# %%capture
# # %reset -f
# %cd '/content/drive/MyDrive/tinystories/'
# !pip -q install wandb torchinfo
#
# # to make torch.compile work on Google Colab
# !export LC_ALL="en_US.UTF-8"
# !export LD_LIBRARY_PATH="/usr/lib64-nvidia"
# !export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
# !ldconfig /usr/lib64-nvidia
# # import os
# # os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # before importing torch!
# # !export CUDA_LAUNCH_BLOCKING=1
# # !export TORCH_USE_CUDA_DSA=1
# # import os
# # os.environ['TORCH_LOGS'] = "+dynamo"
# # os.environ['TORCHDYNAMO_VERBOSE'] = "1"


# todo: why is inference_mode not working with rotary embeddings?
# https://github.com/facebookresearch/xformers/blob/042abc8aa47d1f5bcc2e82df041811de218924ba/xformers/components/positional_embedding/rotary.py#L59

import pickle
import os
import sys
import random
from torchinfo import summary
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.amp import autocast
from sortedcontainers import SortedKeyList
from contextlib import nullcontext
import copy
import time
# from torch.profiler import profile, record_function, ProfilerActivity
# future: ability to attend to nothing, since sequences don't always start with EOS token
# import matplotlib.pyplot as plt

# torch.manual_seed(1337)

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'  # for later use in torch.autocast
wandb_enabled = device_type == 'cuda'
seq_len = 1024# if device_type == 'cuda' else 256
batch_size = 32# if device_type == 'cuda' else 2
depth = 6# if device_type == 'cuda' else 1
attention_heads = 6
d_model = 384
learning_rate = 2e-3
timesteps = 1600 if device_type == 'cuda' else 5
beta1 = 0.9
beta2 = 0.95
eval_interval = 400

grad_clip = 1.0  # todo
device = torch.device(device_type)
torchcompile = True
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
amp_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

with open('cleaned_data.pkl', 'rb') as file:
	data_strings = pickle.load(file)

string_count = len(data_strings)
lengths = [len(d) for d in data_strings]

total_chars = sum(lengths)
print("stories", string_count)
print("chars", total_chars)
print("longest", max(lengths))
print("chars used:", timesteps * seq_len * batch_size)
print("dataset proportion used:", timesteps * seq_len * batch_size / total_chars)


def persist(function):
	def new_func(*args):
		filename = function.__name__ + "_".join([str(a) for a in args]) + ".pkl"
		if os.path.isfile(filename):
			with open(filename, 'rb') as file:
				o = pickle.load(file)
		else:
			print("generating file", filename)
			o = function(*args)
			pickle.dump(o, open(filename, 'wb'))
		return o
	return new_func

def persist_at(filename):
	def decorator(function):
		def new_func(*args):
			nonlocal filename
			filename = filename + ".pkl"
			if os.path.isfile(filename):
				with open(filename, 'rb') as file:
					o = pickle.load(file)
			else:
				print("generating file", filename)
				o = function(*args)
				pickle.dump(o, open(filename, 'wb'))
			return o
		return new_func
	return decorator

@persist
def get_chars():
	all_chars = set()
	for i in data_strings:
		all_chars = all_chars.union(set(i))
	all_chars = sorted(list(all_chars))
	return all_chars
all_chars = get_chars()

print(len(all_chars), "chars used:", ''.join(all_chars))
all_chars_plus_tokens = copy.copy(all_chars)
all_chars_plus_tokens += ['@', '_'] # eos and pad
vocab_size = len(all_chars) + 2  # +1, for EOS. +1, for pad_value
eos_token = vocab_size - 2
pad_token = vocab_size - 1

stoi = {ch: i for i, ch in enumerate(all_chars)}
itos = {i: ch for i, ch in enumerate(all_chars_plus_tokens)}

def encode(s):
	return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(l):
	return ''.join([itos[i] for i in l.tolist()])  # decoder: take a list of integers, output a string



# needs to return: what each sequence maps to
# I think: let's not create a copy with everything in place. because it'll be different during generation
# if we did, we would need to store mask info

@persist_at("packed" + str(seq_len))
def first_fit_decreasing_algorithm(sizes):
	sorted_sizes = [s for s in enumerate(lengths)]  # string index, string size
	sorted_sizes.sort(key=lambda s: s[1])  # in ascending order

	list_of_bins = SortedKeyList(key=lambda s: s[0])  # [remaining space, [(string index, start index, end index)]]
	# remaining space: if = seq_len, it's empty
	# otherwise, it's the size of [sequence plus its eos token]. which can be -1.

	while sorted_sizes:
		biggest = sorted_sizes.pop()  # string index, string size
		if biggest[1] == seq_len - 1:  # exactly fits in one bin, with first and last eos tokens
			list_of_bins.add([0, [(biggest[0], 0, seq_len - 1)]])
		elif biggest[1] > seq_len - 1:  # starting eos token needed
			list_of_bins.add([0, [(biggest[0], 0, seq_len - 1)]])
			index_start = seq_len - 1
			while biggest[1] - index_start >= seq_len:
				list_of_bins.add([0, [(biggest[0], index_start, index_start + seq_len)]])
				index_start += seq_len
			# the last element must go in its own bin; it can't tolerate a start-of-sequence
			# also, if it's too small, just throw it away. here, we enforce 100 chars at least.
			# future: it'd probably make more sense to just create an overlap and fill an entire bucket.
			# theoretically, you'd also mask out the loss on the repetition to prevent overfitting on the repeat. if it matters.
			if index_start < biggest[1] - 100:
				list_of_bins.add([seq_len - (biggest[1] - index_start + 1), [(biggest[0], index_start, biggest[1])]])
		else:
			smallest_free = list_of_bins.bisect_left([biggest[1]])
			if smallest_free == len(list_of_bins):
				list_of_bins.add([seq_len - biggest[1] - 2, [(biggest[0], 0, biggest[1])]])
			else:
				old_bucket = list_of_bins.pop(smallest_free)
				old_bucket[0] -= biggest[1] + 1  # could be -1. that's valid.
				old_bucket[1].append([biggest[0], 0, biggest[1]])
				list_of_bins.add(old_bucket)
	list_of_bins = list(list_of_bins)
	list_of_bins = [p[1] for p in list_of_bins]
	return list(list_of_bins)


pack_mapping = first_fit_decreasing_algorithm(lengths)
# random.shuffle(pack_mapping) # no need to shuffle if we're not splitting into test and train sets!

# python's histogram is very dangerous; ignore any regular spikes. they're aliasing.
# plt.hist([d[0] for d in pack_mapping], bins=90)
# plt.show()
# plt.hist([len(d) for d in data_strings], bins=90)
# plt.show()


class CustomTextDataset(Dataset):
	def __init__(self, SKL):
		"""input: sorted key list generated by bin-packing algorithm"""
		self.SKL = SKL

	def __len__(self):
		return len(self.SKL)

	def __getitem__(self, idx):
		# add +1: last element is predicted
		# torch's crossentropy loss doesn't accept int16! or int32! int64 required
		newtensor = torch.empty(seq_len + 1, dtype=torch.long)

		p = 0  # p = index in the tensor we're constructing
		for i in self.SKL[idx]:
			# i = tuple of (string index, start index, end index)
			L = i[2] - i[1]  # length of the string we're appending
			if p == 0 and i[1] == 0:
				# EOS at beginning and end
				newtensor[0] = eos_token
				newtensor[1:1 + L] = encode(data_strings[i[0]][i[1]:i[2]])
				newtensor[1 + L] = eos_token
				p = 2 + L
			else:
				# EOS at end but not beginning
				newtensor[p:p + L] = encode(data_strings[i[0]][i[1]:i[2]])
				newtensor[p + L] = eos_token
				p += L + 1
		if p < seq_len + 1:
			newtensor[p:].fill_(pad_token)
		return newtensor


# split = int(0.95 * len(pack_mapping))
# split = len(pack_mapping) - batch_size
# train_dataset = CustomTextDataset(pack_mapping[:split])
# test_dataset = CustomTextDataset(pack_mapping[split:])
train_dataset = CustomTextDataset(pack_mapping)

# we already shuffled it. we have to do so before dividing into train and test
# note that our division kind of sucks: many sequences are split across both
train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

# eps = 0 in layernorm? todo, test
# elementwise_affine = False

# from xformers import components
# from xformers.components.positional_embedding import RotaryEmbedding
from rotary import RotaryEmbedding

# copied from x-transformers
def top_p(logits: torch.Tensor, thres = 0.9):
	sorted_logits, sorted_indices = torch.sort(logits, descending = True)
	cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim = -1)

	sorted_indices_to_remove = cum_probs > thres
	sorted_indices_to_remove = F.pad(sorted_indices_to_remove, (1, -1), value = False)

	sorted_logits[sorted_indices_to_remove] = float('-inf')
	return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# from x-transformers
def exists(val):
	return val is not None

def init_zero_(layer):
	nn.init.constant_(layer.weight, 0.)
	if exists(layer.bias):
		nn.init.constant_(layer.bias, 0.)


class Attention(nn.Module):
	def __init__(self, d_model, heads):
		super().__init__()
		self.heads = heads
		self.d_model = d_model
		self.d_k = d_model // heads
		self.W_in = nn.Linear(d_model, 3 * d_model, bias=False)
		self.W_o = nn.Linear(d_model, d_model, bias=False)
		# self.attention = nn.MultiheadAttention(d_model, heads, bias=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None)
		# init_zero_(self.W_o)

	def forward(self, x):
		L = x.size(1)
		# Q shape: (batch_size, seqlen, d_model)
		x = F.layer_norm(x, (self.d_model,), eps=1e-8)
		q, k, v = self.W_in(x).chunk(3, -1)
		# this view and transposition is fine - at least it's done by these three sources:
		# Karpathy's nanoGPT
		# https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
		# https://github.com/facebookresearch/xformers/blob/042abc8aa47d1f5bcc2e82df041811de218924ba/xformers/components/multi_head_dispatch.py#L46
		q = q.view(-1, L, self.heads, self.d_k).transpose(1, 2)
		k = k.view(-1, L, self.heads, self.d_k).transpose(1, 2)
		v = v.view(-1, L, self.heads, self.d_k).transpose(1, 2)
		q, k = rotary_embedding(q=q, k=k) # todo: I wonder if applying this before the QKV reshape improves performance...
		x = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
		x = x.transpose(1, 2).view(-1, L, self.d_model)
		x = self.W_o(x)
		# https://pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html
		return x

class MLP(nn.Module):
	def __init__(self, d_model):
		super().__init__()
		self.c_fc = nn.Linear(d_model, 4 * d_model, bias=False)
		self.c_proj = nn.Linear(4 * d_model, d_model, bias=False)
		self.d_model = d_model

	def forward(self, x):
		x = F.layer_norm(x, (self.d_model,), eps=1e-8)
		x = self.c_fc(x)
		x = F.gelu(x)
		return self.c_proj(x)


class Block(nn.Module):
	def __init__(self, d_model, heads):
		super().__init__()
		self.attn = Attention(d_model, heads)
		self.mlp = MLP(d_model)

	def forward(self, x):
		x = x + self.attn(x)
		x = x + self.mlp(x)
		return x


class Transformer(nn.Module):
	def __init__(self):
		super().__init__()
		self.vocab_size = vocab_size
		self.max_seq_len = seq_len
		self.d_model = d_model
		self.n_layers = depth
		self.heads = attention_heads
		self.embedding = nn.Embedding(vocab_size, d_model)
		nn.init.kaiming_normal_(self.embedding.weight) # from x-transformers
		# assume padding token is the last element, and never emit it
		self.lm_head = nn.Linear(d_model, vocab_size - 1) # bias True
		self.transformerblocks = nn.ModuleList([Block(d_model, attention_heads) for _ in range(depth)])
		self.rotary_embedding = RotaryEmbedding(d_model // attention_heads)
		# needs to be here in the model, so that it can be moved to and from devices


		# todo, change weight inits
		# self.apply(self._init_weights)
		# for pn, p in self.named_parameters():
		# 	if pn.endswith('c_proj.weight') or pn.endswith('W_out.weight'):
		# 		torch.nn.init.normal_(p, mean=0.0, std=0.02/np.sqrt(2 * depth))

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx, selectone=-1):
		if selectone == -1:
			x = idx[:, :-1]
		else:
			x = idx
		assert x.size(1) <= seq_len

		x = self.embedding(x)
		for block in self.transformerblocks:
			x = block(x)
		x = F.layer_norm(x, (self.d_model,), eps=1e-8)
		if selectone == -1:
			logits = self.lm_head(x)
			targets = idx[:, 1:].reshape(-1)
			logits.view(-1, logits.size(-1))
			loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets, ignore_index=pad_token)
			return loss
		else:
			logits = self.lm_head(x[:, selectone:, :])
			return logits

	@torch.no_grad()
	def generate(self, prompt, new_tokens):
		"""In reality, prompt must be of batch size 1. it can't tolerate ragged sequences.
		Otherwise, we'd have to right-align the sequences, but I can't be bothered"""
		assert prompt.dim() == 1
		index = prompt.size(0) - 1 # index of last element
		if prompt.size(0) < seq_len:
			prompt = F.pad(prompt, (0, seq_len - prompt.size(0)), value=pad_token)
		# print("padded size", prompt.shape)
		for _ in range(new_tokens):
			endsection = prompt
			if prompt.size(0) > seq_len:
				endsection = prompt[-seq_len:]
			# print("endsection shape", endsection.shape, index)
			logits = self(endsection[None, :], index)[0, ...]
			logits = top_p(logits)[0, :]
			probs = F.softmax(logits, dim=0)
			choice = torch.multinomial(probs, num_samples=1)
			if index == seq_len - 1:
				prompt = F.pad(prompt, (-1, 1), value=pad_token)
				prompt[-1] = choice
				pass
			else:
				prompt[index + 1] = choice
				index += 1
			# prompt = torch.cat((prompt, choice)) # can't do this, or torch complains! fixed tensor shapes needed

		return prompt


model = Transformer()
# print("model", model)
rotary_embedding = model.rotary_embedding
# ff_glu=True, ff_mult=2.625

torchinfosummary = summary(model, input_data=[train_dataset[0][None, :-1], 0], verbose=1, depth=50)
# if 'google.colab' in sys.modules:
# 	print(torchinfosummary) # before the autoregressive wrapper, or we need to change its shape to +1 seq length
# verbose=2 produces junk: the weights are duplicated over and over

for name, W in model.named_parameters():
	print(name, [a.item() for a in torch.std_mean(W)])

if wandb_enabled:
	# %cd '/'
	import wandb
	wandb.login()
	wandb.init(project="tinystories2", name="from scratch", config={
		"heads": attention_heads,
		"d_model": d_model,
		"depth": depth,
		"batch size": batch_size,
		"seq len": seq_len,
		"timesteps": timesteps,
		"summary": torchinfosummary
	})


model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), betas=(beta1, beta2), lr=learning_rate, weight_decay=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device_type != 'cpu'))

if torchcompile:
	model.compile()

def lr_func(timestep):
	frac = timestep / timesteps
	if frac < 0.25:
		return frac * 4
	else:
		return 1 - (frac - 0.25) / (1 - 0.25)


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

# def print_perf():
# 	from torch.utils.flop_counter import FlopCounterMode
# 	model.train()
# 	random_train_element = random.choice(train_dataset)[:-1]
# 	print("train example", len(random_train_element), random_train_element[:-1], decode(random_train_element))
# 	random_train_element = random_train_element.to(device)
# 	flop_counter = FlopCounterMode(display=False)
# 	with flop_counter:
# 		with amp_ctx:
# 			model(random_train_element).backward()
# 	total_flops = flop_counter.get_total_flops()
# 	print("total flops:", total_flops)

# doesn't work. InternalTorchDynamoError: list index out of range
# if device_type == 'cuda':
# 	print_perf()

@torch.no_grad()
def generate_sample():
	# return # disable, for now
	model.eval()
	# sample_gen_input = next(iter(train_loader)).to(device)
	# result = model.generate(sample_gen_input[0,:seq_len // 4], new_tokens=1024)
	result = model.generate(torch.tensor([eos_token], dtype=torch.long, device=device), new_tokens=seq_len)
	print(decode(result))
	model.train()
#
# @torch.no_grad()
# def test():
# 	model.eval()
# 	total_loss = 0
# 	for idx, inputs in enumerate(test_loader):
# 		# print([decode(r) for r in inputs])
# 		inputs = inputs.to(device)
# 		loss = model(inputs)
# 		total_loss += loss.item()
# 		# print("test loss", total_loss)
# 	print("total loss", total_loss / len(test_loader))
# 	model.train()

def train(epoch):
	model.train()
	for idx, inputs in enumerate(train_loader):
		if idx == timesteps:
			generate_sample()
			return
		# print([decode(r) for r in inputs])
		inputs = inputs.to(device)
		with amp_ctx:
			loss = model(inputs)
		scaler.scale(loss).backward()
		scaler.unscale_(optimizer)
		torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
		scaler.step(optimizer)
		optimizer.zero_grad(set_to_none=True)
		# seems we can zero the grads before the scaler update, even though the have the opposite order at https://pytorch.org/docs/stable/notes/amp_examples.html
		# _check_inf_per_device captures the relevant optimizer state, and that is in scaler.step()
		scaler.update()
		scheduler.step()
		if idx <= 1 and epoch == 0: # some torch.compile thing going on probably
			time_now = time.time_ns()
		time_new = time.time_ns()
		if wandb_enabled:
			wandb.log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0], "time": time_new - time_now})
		time_now = time_new
		if idx == 0 and epoch == 0:
			generate_sample() # ensure it won't OOM later
			time_now = time.time_ns() # avoid crazy upward spikes
		if idx % eval_interval == eval_interval - 1:
			generate_sample()
			time_now = time.time_ns() # avoid crazy upward spikes

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, use_cuda=True) as prof:
# 	with record_function("model_inference"):
# 		train(0)
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
train(0)

if device_type == 'cuda':
	for name, W in model.named_parameters():
		print(name, [a.item() for a in torch.std_mean(W)])

if wandb_enabled:
	wandb.finish()
import sys
if 'google.colab' in sys.modules:
	from google.colab import runtime
	runtime.unassign()