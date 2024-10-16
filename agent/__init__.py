import abc
import torch
from pathlib import Path
import numpy as np
from collections import deque


class ObservationBuffer:
    def __init__(self, capacity: int, dump_every: int = 0, fn=None):

        self.capacity = capacity
        self.dump_every = dump_every
        self.fn = fn

        self.no_waiting = fn is None

        self.buffer = deque(maxlen=capacity)
        self.waiting = []

    def __len__(self):
        return len(self.buffer) + len(self.waiting)

    def __dump(self):
        assert not self.no_waiting, "No dump function provided"
        assert len(self.waiting) > 0, "Nothing to dump"
        waiting_batch = np.stack(self.waiting)
        processed_batch = self.fn(waiting_batch)
        self.buffer.extend(processed_batch.tolist())
        self.waiting.clear()

    def add(self, obs):
        if self.no_waiting:
            self.buffer.append(obs)
        else:
            self.waiting.append(obs)
            if len(self.waiting) >= self.dump_every:
                self.__dump()

    def sample(self, n):
        if len(self.buffer) == 0:
            assert len(self.waiting) > 0, "Nothing to sample"
            self.__dump()
        indices = np.random.randint(0, len(self.buffer), n)
        return np.array([self.buffer[i] for i in indices])


class Agent(object):
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.device = torch.device(cfg.device)

        self.obs_buffer = None  # need to be initialized in the child class
        self._prepared = False

    def sample_from_obs_buffer(self, n):
        return self.obs_buffer.sample(n)

    def obs_buffer_len(self):
        return len(self.obs_buffer)

    def get_obs_shape(self):
        raise NotImplementedError

    def prepare(self, reward_fn=None):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def run(self, num_train_steps):
        raise NotImplementedError

    def collect_obs(self, num_obs, verbose=False):
        raise NotImplementedError

    def load_agent(self, dir_path):
        raise NotImplementedError

    def save_agent(self, dir_path):
        raise NotImplementedError
