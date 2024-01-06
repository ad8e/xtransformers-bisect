Here, I tried to reproduce an x-transformers model exactly, from scratch. But my model's loss curves are different. See https://wandb.ai/team-ad8e/bisect/workspace?workspace=user-ad8e

The model from scratch is better from timesteps 0-400, then x-transformers is better from 400-600, then the model from scratch is better after 600.

I have not been able to figure out the difference. The mean and std dev for all layers match.

I'm suspicious of the RotaryEmbedding because I haven't checked it yet.

torchinfo claims the models are different, but it's miscounting because of the RotaryEmbedding.

I don't know why my model runs faster.

I glued the QKV matrices together.

To run:

1. download https://huggingface.co/datasets/roneneldan/TinyStories/blob/main/TinyStories_all_data.tar.gz
2. run prepare.py to produce a pickle file
3. run `train x-transformers.py` to test the x-transformers model, or `train xformers.py` to test the xformers model. I left a `wandb.login()` in there so you'll have to either register in wandb, or set `wandb_enabled = False`.

I don't expect anyone to do this testing; maybe I'll get back to it some day.

Two example runs are in `arch xformers.txt` and `arch xtransformers.txt`, but I'm not sure they're up to date.