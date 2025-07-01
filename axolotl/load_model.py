import os
import torch
from .model.transformer import DiscreteDiT
from .model.ema import ExponentialMovingAverage
import numpy as np

from . import (
    utils,
    graph_lib,
    noise_lib
)

from omegaconf import OmegaConf

def load_model_hf(dir, device):
    score_model = DiscreteDiT.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device):
    config = utils.load_hydra_config_from_run(root_dir)
    graph = graph_lib.get_graph(config, device)
    noise = noise_lib.get_noise(config.noise.type, config.noise.sigma_min, config.noise.sigma_max, config.noise.eps).to(device)
    score_model = DiscreteDiT(config).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=config.training.ema)

    ckpt_dir = os.path.join(root_dir, "checkpoints-meta", "checkpoint.pth")
    
    loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False) # TODO safe serialization when saving, to do weights_only=True

    score_model.load_state_dict(loaded_state['model'])
    ema.load_state_dict(loaded_state['ema'])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model(root_dir, device):
    try:
        return load_model_hf(root_dir, device)
    except:
        return load_model_local(root_dir, device)