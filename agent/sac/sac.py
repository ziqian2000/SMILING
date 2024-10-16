import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import utils

import hydra

from agent.sac.actor import DiagGaussianActor
from agent.sac.critic import DoubleQCritic


# class Agent(object):
#     def reset(self):
#         """For state-full agents this function performs reseting at the beginning of each episode."""
#         pass

#     @abc.abstractmethod
#     def train(self, training=True):
#         """Sets the agent in either training or evaluation mode."""

#     @abc.abstractmethod
#     def update(self, replay_buffer, logger, step):
#         """Main function of the agent that performs learning."""

#     @abc.abstractmethod
#     def act(self, obs, sample=False):
#         """Issues an action given an observation."""


class SACAlgorithm:
    """SAC algorithm."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        action_range,
        device,
        critic_cfg,
        actor_cfg,
        discount,
        init_temperature,
        alpha_lr,
        alpha_betas,
        actor_lr,
        actor_betas,
        actor_update_frequency,
        critic_lr,
        critic_betas,
        critic_tau,
        critic_target_update_frequency,
        batch_size,
        learnable_temperature,
        num_seed_steps,
    ):
        # super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature

        self.critic = DoubleQCritic(**critic_cfg.params).to(self.device)
        self.critic_target = DoubleQCritic(**critic_cfg.params).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = DiagGaussianActor(**actor_cfg.params).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=actor_betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=critic_betas
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=alpha_betas
        )

        self.train()
        self.critic_target.train()

    def reset(self):
        """For state-full agents this function performs reseting at the beginning of each episode."""
        pass

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        assert len(action.shape) == 2 and action.shape[0] == 1
        assert action.ndim == 2 and action.shape[0] == 1
        return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        metrics = dict()

        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q
        )
        metrics["train_critic/loss"] = critic_loss

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(logger, step)

        return metrics

    def update_actor_and_alpha(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        metrics["train_actor/loss"] = actor_loss
        metrics["train_actor/target_entropy"] = self.target_entropy
        metrics["train_actor/entropy"] = -log_prob.mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (
                self.alpha * (-log_prob - self.target_entropy).detach()
            ).mean()
            metrics["train_alpha/loss"] = alpha_loss
            metrics["train_alpha/value"] = self.alpha
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        return metrics

    def update(self, replay_buffer, step, reward_fn, use_action):
        metrics = dict()

        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size
        )

        if reward_fn is not None:
            _reward = reward_fn(
                obs if not use_action else torch.cat([obs, action], dim=-1)
            )
            assert _reward.shape == reward.shape
            reward = _reward
            metrics.update(
                {
                    "rl_reward/mean": reward.mean().item(),
                    "rl_reward/max": reward.max().item(),
                    "rl_reward/min": reward.min().item(),
                    "rl_reward/std": reward.std().item(),
                }
            )

        metrics["train/batch_reward"] = reward.mean()

        metrics.update(
            self.update_critic(obs, action, reward, next_obs, not_done_no_max, step)
        )

        if step % self.actor_update_frequency == 0:
            metrics.update(self.update_actor_and_alpha(obs, step))

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return metrics

    def bc_update(self, obs, action, loss_type: str):
        metrics = dict()
        dist = self.actor(obs)
        if loss_type == "mle":
            # clip action
            action = torch.clamp(action, -1 + 1e-5, 1 - 1e-5)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            loss = -log_prob.mean()
        elif loss_type == "mse":
            loss = F.mse_loss(dist.mean, action)
        else:
            raise ValueError(f"loss_type {loss_type} not supported")
        metrics["loss"] = loss.item()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return metrics

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), model_dir / "actor.pt")
        torch.save(self.critic.state_dict(), model_dir / "critic.pt")

    def load(self, model_dir):
        self.actor.load_state_dict(torch.load(model_dir / "actor.pt"))
        self.critic.load_state_dict(torch.load(model_dir / "critic.pt"))
