import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from algorithm import Algorithm


class DAC(Algorithm):
    def __init__(self, cfg, obs_dim):
        super(DAC, self).__init__(cfg, obs_dim)

        assert isinstance(
            obs_dim, int
        ), "DAC only supports 1D observation space for now"

        self.reward_style = cfg.alg.reward_style

        self.linear = cfg.use_linear
        if self.linear:
            print("No activation function in MLP for DAC")

        modules = [
            nn.Linear(obs_dim, cfg.alg.hidden_dim),
            nn.LeakyReLU() if not self.linear else nn.Identity(),
        ]
        for _ in range(cfg.alg.hidden_depth - 1):
            modules.append(nn.Linear(cfg.alg.hidden_dim, cfg.alg.hidden_dim))
            modules.append(nn.LeakyReLU() if not self.linear else nn.Identity())
        modules.append(nn.Linear(cfg.alg.hidden_dim, 1))
        self.model = nn.Sequential(*modules).to(self.device)

        # self.model = nn.Sequential(
        #     nn.Linear(obs_dim, cfg.alg.hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(cfg.alg.hidden_dim, 1),
        # ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=cfg.alg.lr)

        self.normalize_reward = cfg.alg.normalize_reward
        self.target_std = cfg.alg.target_std

    def compute_gan_loss(self, policy_output, expert_output):

        assert len(policy_output.shape) == 2 and len(expert_output.shape) == 2
        assert policy_output.shape[1] == 1 and expert_output.shape[1] == 1

        zeros = torch.zeros(expert_output.shape[0], 1).to(self.device)
        ones = torch.ones(policy_output.shape[0], 1).to(self.device)

        loss_policy = F.binary_cross_entropy_with_logits(
            policy_output, zeros, reduction="mean"
        )
        loss_expert = F.binary_cross_entropy_with_logits(
            expert_output, ones, reduction="mean"
        )

        loss = (loss_policy + loss_expert) / 2

        return loss

    def update(
        self,
        dataloader,
        val_dataloader,
        num_epochs,
        verbose: bool,
        log: bool,
        global_epoch: int,
    ):
        for epoch in range(num_epochs):

            if verbose:
                progress_bar = tqdm(total=len(dataloader))
                progress_bar.set_description(f"Epoch {epoch}")

            metrics = dict()
            metrics.update({"dac/global_epoch": global_epoch})

            for _, batch in enumerate(dataloader):

                batch = batch.float().to(self.device)

                policy_obs = batch[:, : batch.shape[1] // 2]
                expert_obs = batch[:, batch.shape[1] // 2 :]

                # GAN loss
                policy_output = self.model(policy_obs)
                expert_output = self.model(expert_obs)
                gan_loss = self.compute_gan_loss(policy_output, expert_output)

                # grad penalty
                alpha = torch.rand(policy_obs.shape[0], 1).to(self.device)
                interpolated = alpha * policy_obs + (1 - alpha) * expert_obs
                interpolated.requires_grad = True
                output = self.model(interpolated)
                grad = torch.autograd.grad(
                    outputs=output,
                    inputs=interpolated,
                    grad_outputs=torch.ones_like(output),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                assert len(grad.shape) == 2
                grad_penalty = ((grad.norm(2, dim=1) - 1) ** 2).mean()

                # compute and optimize loss
                loss = gan_loss + 10 * grad_penalty

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logs = {
                    "dac/loss": loss.item(),
                    "dac/gan_loss": gan_loss.item(),
                    "dac/grad_penalty": grad_penalty.item(),
                }

                if verbose:
                    progress_bar.update(1)
                    progress_bar.set_postfix(**logs)

            if verbose:
                progress_bar.close()

            metrics.update(logs)  # only log last batch
            if log:
                wandb.log(metrics)

    @torch.no_grad()
    def reward_fn(self, obs):

        assert (
            len(obs.shape) == 2 or len(obs.shape) == 4
        ), obs.shape  # [bs, obs_dim] or [bs, c, h, w]

        if len(obs.shape) == 4:
            assert False, "does not support image input"

        if self.reward_style == "airl":
            r = self.model(obs)
        elif self.reward_style == "gail":
            r = -torch.log(1.0 - torch.sigmoid(self.model(obs)) + 1e-8)
        else:
            raise NotImplementedError

        if self.normalize_reward:
            r = (r - r.mean()) / (r.std() + 1e-6) * self.target_std

        return r
