import abc
import torch
from pathlib import Path
import numpy as np
from collections import deque


class Algorithm:
    def __init__(self, cfg, obs_dim):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.device = torch.device(cfg.device)

    def update(self):
        raise NotImplementedError

    def reward_fn(self, obs):
        raise NotImplementedError
