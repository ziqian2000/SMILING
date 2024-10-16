import math
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from diffusers import DDPMScheduler
from algorithm import Algorithm


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(
        self, state_dim, device, time_emb_dim, hidden_dim, hidden_depth, no_activation
    ):

        if no_activation:
            print("No activation function in MLP for diffusion model")

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 2),
            nn.LeakyReLU() if not no_activation else nn.Identity(),
            nn.Linear(time_emb_dim * 2, time_emb_dim),
        )

        input_dim = state_dim + time_emb_dim

        modules = [
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU() if not no_activation else nn.Identity(),
        ]
        for _ in range(hidden_depth - 1):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU() if not no_activation else nn.Identity())
        self.mid_layer = nn.Sequential(*modules)

        self.final_layer = nn.Linear(hidden_dim, state_dim)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = torch.cat([x, t], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)


class DiffusionModel:

    def __init__(
        self,
        obs_dim,
        timesteps,
        lr,
        time_emb_dim,
        hidden_dim,
        hidden_depth,
        linear=False,
    ):

        self.device = torch.device("cuda")

        assert isinstance(obs_dim, tuple) or isinstance(obs_dim, int)

        self.pixel_input = isinstance(obs_dim, tuple)
        assert not self.pixel_input, "does not support pixel input"

        print("diffusion input size", obs_dim)

        self.state_dim = obs_dim

        self.model = MLP(
            state_dim=obs_dim,
            device=self.device,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            no_activation=linear,
        ).to(self.device)

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=timesteps)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def preprocess_clean_images(self, batch):
        obs = batch.float().to(self.device).view(batch.shape[0], self.state_dim)
        assert len(obs.shape) == 2
        # has to be [BATCH_SIZE, N_OBSERVATIONS])
        return obs

    def train(
        self,
        dataloader,
        val_dataloader,
        num_epochs,
        verbose: bool,
        log: bool,
        global_epoch: int,
    ):

        # Now you train the model
        for epoch in range(num_epochs):

            # if verbose:
            #     progress_bar = tqdm(total=len(dataloader))
            #     progress_bar.set_description(f"Epoch {epoch}")

            metrics = dict()
            metrics.update({"diffusion/global_epoch": global_epoch})

            train_loss = 0
            for _, batch in enumerate(dataloader):

                clean_images = self.preprocess_clean_images(batch)

                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape, device=clean_images.device)
                batch_size = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    self.noise_scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=clean_images.device,
                    dtype=torch.int64,
                )

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(
                    clean_images, noise, timesteps
                )

                # Predict the noise residual
                noise_pred = self.model(noisy_images, timesteps)
                loss = F.mse_loss(noise_pred, noise)
                train_loss += loss.detach().item() * batch_size

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                logs = {
                    "diffusion/loss": loss.detach().item(),
                    "diffusion/lr": self.optimizer.param_groups[0]["lr"],
                }

                # if verbose:
                #     progress_bar.update(1)
                #     progress_bar.set_postfix(**logs)

            train_loss /= len(dataloader.dataset)

            if verbose and epoch % 5 == 0:
                print(f"Epoch {epoch} : train_loss {train_loss}")

            # if verbose:
            #     progress_bar.close()

            if val_dataloader is not None and epoch % 50 == 0:
                with torch.no_grad():
                    val_loss = 0
                    for _, batch in enumerate(val_dataloader):

                        clean_images = self.preprocess_clean_images(batch)

                        noise = torch.randn(
                            clean_images.shape, device=clean_images.device
                        )
                        batch_size = clean_images.shape[0]
                        timesteps = torch.randint(
                            0,
                            self.noise_scheduler.config.num_train_timesteps,
                            (batch_size,),
                            device=clean_images.device,
                            dtype=torch.int64,
                        )
                        noisy_images = self.noise_scheduler.add_noise(
                            clean_images, noise, timesteps
                        )
                        noise_pred = self.model(noisy_images, timesteps)
                        val_loss += (
                            F.mse_loss(noise_pred, noise).detach().item() * batch_size
                        )

                    val_loss /= len(val_dataloader.dataset)

                    if verbose:
                        print(f"val_loss: {val_loss}")

                    metrics.update({"diffusion/val_loss": val_loss})

            metrics.update(logs)  # only log last batch
            if log:
                wandb.log(metrics)

    def save(self, model_dir):
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), model_dir / "diffusion.pt")
        print(f"diffusion model saved at {model_dir}", flush=True)

    def load(self, model_dir):
        self.model.load_state_dict(torch.load(model_dir / "diffusion.pt"))
        print(f"diffusion model loaded from {model_dir}", flush=True)


class Diff(Algorithm):

    def __init__(self, cfg, obs_dim):
        super(Diff, self).__init__(cfg, obs_dim)

        self.expert_diff = DiffusionModel(
            obs_dim=obs_dim,
            timesteps=cfg.alg.timesteps,
            lr=cfg.alg.lr,
            time_emb_dim=cfg.alg.time_emb_dim,
            hidden_dim=cfg.alg.hidden_dim,
            hidden_depth=cfg.alg.hidden_depth,
            linear=cfg.use_linear,
        )

        self.current_diff = DiffusionModel(
            obs_dim=obs_dim,
            timesteps=cfg.alg.timesteps,
            lr=cfg.alg.lr,
            time_emb_dim=cfg.alg.time_emb_dim,
            hidden_dim=cfg.alg.hidden_dim,
            hidden_depth=cfg.alg.hidden_depth,
            linear=cfg.use_linear,
        )

        self.normalize_reward = cfg.alg.normalize_reward
        self.target_std = cfg.alg.target_std

    def load_expert_diff(self, diff_model_name):
        self.expert_diff.load(Path.cwd() / "models" / diff_model_name)

    def save_expert_diff(self, diff_model_name):
        self.expert_diff.save(Path.cwd() / "models" / diff_model_name)

    def update(
        self,
        dataloader,
        val_dataloader,
        num_epochs: int,
        verbose: bool,
        log: bool,
        global_epoch: int,
        train_expert: bool = False,
    ):
        diff = self.expert_diff if train_expert else self.current_diff
        diff.train(
            dataloader=dataloader,
            val_dataloader=val_dataloader,
            num_epochs=num_epochs,
            verbose=verbose,
            log=log,
            global_epoch=global_epoch,
        )

    @torch.no_grad()
    def __reward_compute(
        self,
        obs,
        t_sample_cnt,
    ):

        assert (
            len(obs.shape) == 2 or len(obs.shape) == 4
        ), obs.shape  # [bs, obs_dim] or [bs, c, h, w]

        assert (
            self.expert_diff.noise_scheduler.config.num_train_timesteps % t_sample_cnt
            == 0
        )

        batch_size = obs.shape[0]

        if len(obs.shape) == 2:
            t_obs = obs.repeat(t_sample_cnt, 1)  # [bs * t_sample_cnt, obs_dim]
        else:
            obs = self.expert_diff.preprocess_clean_images(obs)
            t_obs = obs.repeat(t_sample_cnt, 1, 1, 1)  # [bs * t_sample_cnt, c, h, w]

        timesteps = torch.arange(
            0,
            self.expert_diff.noise_scheduler.config.num_train_timesteps,
            self.expert_diff.noise_scheduler.config.num_train_timesteps // t_sample_cnt,
            device=obs.device,
            dtype=torch.int64,
        ).repeat(
            batch_size
        )  # [bs * t_sample_cnt]

        noise = torch.randn(t_obs.shape, device=obs.device)
        noisy_obs = self.expert_diff.noise_scheduler.add_noise(t_obs, noise, timesteps)

        expert_pred = self.expert_diff.model(noisy_obs, timesteps)
        current_pred = self.current_diff.model(noisy_obs, timesteps)

        assert expert_pred.shape == t_obs.shape

        if len(obs.shape) == 2:
            term1 = torch.mean((expert_pred - noise) ** 2, axis=1, keepdim=True)
            term2 = torch.mean((current_pred - noise) ** 2, axis=1, keepdim=True)
        else:
            term1 = torch.mean((expert_pred - noise) ** 2, axis=(1, 2, 3)).view(-1, 1)
            term2 = torch.mean((current_pred - noise) ** 2, axis=(1, 2, 3)).view(-1, 1)

        assert len(term1.shape) == 2

        reward = term2 - term1

        reward = reward.view(t_sample_cnt, batch_size).mean(dim=0)  # [bs]
        reward = reward.unsqueeze(1)  # [bs, 1]

        assert len(reward.shape) == 2 and reward.shape[0] == batch_size

        return reward

    @torch.no_grad()
    def reward_fn(self, obs):
        r = self.__reward_compute(
            obs,
            self.cfg.alg.t_sample_cnt,
        )

        # normalize r
        if self.normalize_reward:
            r = (r - r.mean()) / (r.std() + 1e-6) * self.target_std

        return r
