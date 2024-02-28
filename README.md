# Score Entropy Discrete Diffusion
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repo contains a PyTorch implementation for the paper [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution
](https://arxiv.org/abs/2310.16834) by [Aaron Lou](https://aaronlou.com), [Chenlin Meng](https://cs.stanford.edu/~chenlin/) and [Stefano Ermon](https://cs.stanford.edu/~ermon/).

![cover](assets/main.gif)

## Design Choices

This codebase is built modularly to promote future research (as opposed to a more compact framework, which would be better for applications). The primary files are 

1. ```noise_lib.py```: the noise schedule
2. ```graph_lib```: the forward diffusion process
3. ```sampling.py```: the sampling strategies
4. ```model/```: the model architecture

## Installation

Simply run

```
conda env create -f environment.yml
```

which will create a ```sedd``` environment with packages installed. Note that this installs with CUDA 11.8, and different CUDA versions must be installed manually. The biggest factor is making sure that the ```torch``` and ```flash-attn``` packages use the same CUDA version (more found [here](https://github.com/Dao-AILab/flash-attention)).

## Working with Pretrained Models

### Download Models

Our pretrained models are hosted on huggingface ([small](https://huggingface.co/louaaron/sedd-small), [medium](https://huggingface.co/louaaron/sedd-medium)). However, models can also be loaded in locally (say after training). All functionality is found in ```load_model.py```.

```
# load in a pretrained model
pretrained_small_model, graph, noise = load_model("louaaron/sedd-small")
pretrained_medium_model, graph, noise = load_model("louaaron/sedd-medium")
# load in a local experiment
local_model, graph, noise = load_model("exp_local/experiment)
```

This loading gives the model, as well as the graph and noise (which are used for the loss/sampling setup).

### Run Sampling

We can run sampling using a command 

```
python run_sample.py --model_path MODEL_PATH --steps STEPS
```

We can also sample conditionally using

```
python run_sample_cond.py --model_path MODEL_PATH --step STEPS --prefix PREFIX --suffix SUFFIX
```

## Training New Models

### Run Training

We provide training code, which can be run with the command
```
python run_train.py
```
This creates a new directory `direc=exp_local/DATE/TIME` with the following structure (compatible with running sampling experiments locally)
```
├── direc
│   ├── .hydra
│   │   ├── config.yaml
│   │   ├── ...
│   ├── checkpoints
│   │   ├── checkpoint_*.pth
│   ├── checkpoints-meta
│   │   ├── checkpoint.pth
│   ├── samples
│   │   ├── iter_*
│   │   │   ├── sample_*.txt
│   ├── logs
```
Here, `checkpoints-meta` is used for reloading the run following interruptions, `samples` contains generated images as the run progresses, and `logs` contains the run output. Arguments can be added with `ARG_NAME=ARG_VALUE`, with important ones being:
```
ngpus                     the number of gpus to use in training (using pytorch DDP)
training.accum            number of accumulation steps, set to 1 for small and 2 for medium (assuming an 8x80GB node)
noise.type                one of geometric, loglinear 
graph.type                one of uniform, absorb
model                     one of small, medium
model.scale_by_sigma      set to False if graph.type=uniform (not yet configured)
```
Some example commands include
```
# training hyperparameters for SEDD absorb
python train.py noise_lib=loglinear graph.type=absorb model=medium training.accum=2
# training hyperparameters for SEDD uniform
python train.py noise_lib=geometric graph.type=uniform model=small model.scale_by_sigma=False
```

## Other Features

### SLURM compatibility

To train on slurm, simply run 
```
python train.py -m args
```

## Citation
```
@article{lou2024discrete,
  title={Discrete diffusion modeling by estimating the ratios of the data distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```
## Acknowledgements

This repository builds heavily off of [score sde](https://github.com/yang-song/score_sde_pytorch), [plaid](https://github.com/igul222/plaid), and [DiT](https://github.com/facebookresearch/DiT).