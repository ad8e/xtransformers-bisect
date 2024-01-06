wandb_enabled = 1

# import sys
# IN_COLAB = 'google.colab' in sys.modules
# directory = ''
# if IN_COLAB:
# 	%reset -f
# 	%cd '/content/drive/MyDrive/tinystories/'
# 	!pip -q install wandb einops torchinfo
#
# 	# to make torch.compile work on Google Colab
# 	!export LC_ALL="en_US.UTF-8"
# 	!export LD_LIBRARY_PATH="/usr/lib64-nvidia"
# 	!export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
# 	!ldconfig /usr/lib64-nvidia


import pickle
import os
import sys
import random
from torchinfo import summary
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp.grad_scaler import GradScaler
from torch.amp import autocast
from sortedcontainers import SortedKeyList
from contextlib import nullcontext
from x_transformers import TransformerWrapper, Decoder
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper, top_p
import copy
import time
# from torch.profiler import profile, record_function, ProfilerActivity
# future: ability to attend to nothing, since sequences don't always start with EOS token
# import matplotlib.pyplot as plt

# torch.manual_seed(1337)

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'  # for later use in torch.autocast
seq_len = 1024# if device_type == 'cuda' else 512
batch_size = 32# if device_type == 'cuda' else 1
depth = 6# if device_type == 'cuda' else 1
attention_heads = 6
n_embd = 384
learning_rate = 2e-3
timesteps = 1600 if device_type == 'cuda' else 5
beta1 = 0.9
beta2 = 0.95
eval_interval = 400

grad_clip = 1.0  # todo
device = torch.device(device_type)
compile = True
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
			# print("i", i, "p", p)
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

model = TransformerWrapper(
	num_tokens=vocab_size,
	max_seq_len=seq_len,
	logits_dim=vocab_size - 1,
	attn_layers=Decoder(
		dim=n_embd,
		depth=depth,
		heads=attention_heads,
		rotary_pos_emb=True,
		attn_flash=True,
		ff_no_bias=True,
	)
	# ff_glu=True, ff_mult=2.625
) # no need to set casual=True: it's automatic. x-transformers even complains if you do

torchinfosummary = summary(model, input_data=train_dataset[0][None, :-1], verbose=1, depth=50)
# if 'google.colab' in sys.modules:
# 	print(torchinfosummary) # before the autoregressive wrapper, or we need to change its shape to +1 seq length

if wandb_enabled:
	import wandb
	wandb.login()
	wandb.init(project="tinystories2", name="x-transformers", config={
		"heads": attention_heads,
		"n_embd": n_embd,
		"depth": depth,
		"batch size": batch_size,
		"seq len": seq_len,
		"timesteps": timesteps,
		"summary": torchinfosummary
	})


model = AutoregressiveWrapper(model, ignore_index = pad_token, pad_value = pad_token)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), betas=(beta1, beta2), lr=learning_rate, weight_decay=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16' and device_type != 'cpu'))

for name, W in model.named_parameters():
	print(name, [a.item() for a in torch.std_mean(W)])

if compile:
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

@torch.inference_mode()
def generate_sample():
	model.eval()
	sample_gen_input = next(iter(train_loader)).to(device)
	result = model.generate(sample_gen_input[0:seq_len // 4], seq_len = 1024) # don't set eos_token or it'll stop immediately
	print(decode(result[0, :]))
	model.train()
#
# @torch.inference_mode()
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
		scaler.update()
		scheduler.step()
		if idx == 0 and epoch == 0:
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
if 'google.colab' in sys.modules:
	from google.colab import runtime
	runtime.unassign()