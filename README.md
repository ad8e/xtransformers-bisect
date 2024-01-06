Here, I tried to reproduce an x-transformers model exactly, from scratch. But my model's loss curves are different. See https://wandb.ai/team-ad8e/bisect/workspace?workspace=user-ad8e

The model from scratch is better from timesteps 60-250, where x-transformers has a hump in its loss. First three runs are without data shuffling, and there's very significant ordering effects in the data. Last two runs are with data shuffling.

I have not been able to figure out the reason for the hump. The mean and std dev for all layers match.

The x-transformers model had the scaling ablated from its LayerNorm; you would need to make this change to your own copy of x-transformers to match these results. The from-scratch model also has its scaling removed from LayerNorm.

		class LayerNorm(nn.Module):
		    def __init__(self, dim):
		        """
		        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
		        """
		        super().__init__()

		    def forward(self, x):
		        return F.layer_norm(x, x.shape[-1:])

I'm suspicious of the RotaryEmbedding because I haven't checked it yet.

torchinfo claims the models are different, but it's miscounting because of the RotaryEmbedding.

I don't know why my model runs 10% faster.

I glued the QKV matrices together.

To run:

1. download https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories_all_data.tar.gz
2. run prepare.py to produce a pickle file
3. run `train x-transformers.py` to test the x-transformers model, or `train xformers.py` to test the xformers model. I left a `wandb.login()` in there so you'll have to either register in wandb, or set `wandb_enabled = False`.

I don't expect anyone to do this testing; maybe I'll get back to it some day.

Two example runs are in `arch xformers.txt` and `arch xtransformers.txt`, using shuffled data.