import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import pickle as pkl
import wandb
import hydra
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from collections import namedtuple

from video import VideoRecorder
from logger import Logger
from replay_buffer import ReplayBuffer
import utils

import agent.sac.sac
from agent import Agent, ObservationBuffer


def make_alg(cfg):
    return agent.sac.sac.SACAlgorithm(**cfg.agent.params)


Snapshot = namedtuple(
    "Snapshot", ["obs", "done", "episode_reward", "episode_step", "info"]
)


class SACAgent(Agent):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.env = utils.make_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.alg = make_alg(cfg)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None)
        self.step = 0

        self.save_action = cfg.use_action
        self.obs_buffer = ObservationBuffer(capacity=cfg.obs_buffer_capacity)

        self.snapshot = Snapshot(
            obs=None, done=True, episode_reward=None, episode_step=None, info=dict()
        )

    def get_obs_shape(self):
        return self.env.observation_space.shape[0]

    def get_action_shape(self):
        return self.env.action_space.shape[0]

    def prepare(self, reward_fn=None):
        self.reward_fn = reward_fn
        self._prepared = True

    def evaluate(self, video_prefix=None):
        # for i in range(20):
        #     self.env.reset()

        average_episode_reward = 0
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()[0]
            self.alg.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(self.alg):
                    action = self.alg.act(obs, sample=False)
                    # action = self.env.action_space.sample()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.video_recorder.record(self.env)
                episode_reward += reward

            average_episode_reward += episode_reward
            self.video_recorder.save(
                f"{self.step if video_prefix is None else video_prefix}.mp4"
            )
        average_episode_reward /= self.cfg.num_eval_episodes

        return average_episode_reward

    def run_bc(self, dataset, num_train_epochs):
        for i in range(num_train_epochs):

            metrics = dict()

            if i % 2000 == 0:
                ret = self.evaluate()
                metrics["episode_reward"] = ret
                print(f"eval: {ret}")

            obs = torch.tensor(dataset[:, : self.get_obs_shape()]).to(self.device)
            action = torch.tensor(dataset[:, self.get_obs_shape() :]).to(self.device)

            metrics.update(self.alg.bc_update(obs, action, loss_type=self.cfg.alg.loss))

            if self.cfg.log_wandb:
                wandb.log(metrics)

            if i % 100 == 0:
                print(f"epoch {i}, loss: {metrics['loss']}")

    def run(self, num_train_steps):
        assert self._prepared, "Call prepare() before run()"

        reward_fn = self.reward_fn

        episode = 0

        obs = self.snapshot.obs
        done = self.snapshot.done
        episode_reward = self.snapshot.episode_reward
        episode_step = self.snapshot.episode_step
        info = self.snapshot.info

        start_time = time.time()

        end_step = self.step + num_train_steps
        # init_step = self.step

        while self.step < end_step:
            metrics = dict()
            metrics["train/global_step"] = self.step

            if done:

                if self.step > 0:
                    metrics["train/duration"] = time.time() - start_time
                    start_time = time.time()
                    metrics["train/episode_reward"] = episode_reward
                    episode += 1
                    metrics["train/episode"] = episode

                    print(
                        f"Step: {self.step}, Episode: {episode}, Reward: {episode_reward}"
                    )

                    # evaluate agent periodically
                    if self.step % self.cfg.eval_frequency == 0:
                        metrics["eval/episode"] = episode
                        metrics["eval/episode_reward"] = self.evaluate()

                    if self.step % self.cfg.save_model_every == 0:
                        path = (
                            self.work_dir
                            / "models"
                            / self.cfg.run_name
                            / str(self.step)
                        )
                        path.mkdir(parents=True, exist_ok=True)
                        self.alg.save(path)

                obs = self.env.reset()[0]
                done = False
                episode_reward = 0
                episode_step = 0

                self.alg.reset()

            # sample action for data collection
            if self.step < self.cfg.agent.params.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.alg):
                    action = self.alg.act(obs, sample=True)

            if self.save_action:
                self.obs_buffer.add(np.concatenate([obs, action]))
            else:
                self.obs_buffer.add(obs.copy())

            # run training update
            if self.step >= self.cfg.agent.params.num_seed_steps:
                metrics.update(
                    self.alg.update(
                        self.replay_buffer,
                        self.step,
                        reward_fn,
                        use_action=self.save_action,
                    )
                )

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs

            episode_step += 1
            self.step += 1

            if self.cfg.log_wandb:
                wandb.log(metrics)

        self.snapshot = Snapshot(
            obs=obs,
            done=done,
            episode_reward=episode_reward,
            episode_step=episode_step,
            info=info,
        )

    def collect_obs(self, num_obs, save_action, verbose=False):
        obs_list = []
        episode, done = -1, True

        for i in tqdm(range(num_obs), disable=not verbose, desc="Collecting obs"):
            if done:
                obs = self.env.reset()[0]
                self.alg.reset()
                done = False
                episode += 1

            with utils.eval_mode(self.alg):
                action = self.alg.act(obs, sample=True)
                obs_list.append(
                    obs if not save_action else np.concatenate([obs, action])
                )
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            obs = next_obs

        return obs_list, episode

    def load_agent(self, path):
        self.alg.load(path)
