import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import hydra
import time
import wandb
import re
import jax
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from functools import partial as bind
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import agent.dreamerv3.embodied as embodied
import utils
from agent import Agent, ObservationBuffer


def make_agent(config):
    from agent.dreamerv3.embodied.agents.dreamerv3 import agent as agt

    env = utils.make_env(config, 0)
    agent = agt.Agent(env.obs_space, env.act_space, config.agent)
    env.close()
    return agent


def make_logger(config, env, wandb_enabled, logdir):
    step = embodied.Counter()
    multiplier = config.env.get(env.split("_")[0], {}).get("repeat", 1)
    outputs = [
        embodied.logger.TerminalOutput(config.filter, "Agent"),
        embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
        embodied.logger.JSONLOutput(logdir, "scores.jsonl", "episode/score"),
        # embodied.logger.TensorBoardOutput(
        #     logdir, config.run.log_video_fps, config.tensorboard_videos
        # ),
    ]
    if wandb_enabled:
        wandb_run_name = f"{env}.{config.method}.{config.seed}"
        outputs.append(embodied.logger.WandBOutput(logdir, wandb_run_name, config))
    logger = embodied.Logger(step, outputs, multiplier)
    return logger


def make_replay(config, directory=None, is_eval=False, rate_limit=False):
    directory = directory and embodied.Path(config.logdir) / directory
    size = config.replay_size // 10 if is_eval else config.replay_size
    kwargs = {}
    if rate_limit and config.run.train_ratio > 0:
        kwargs["samples_per_insert"] = config.run.train_ratio / config.batch_length
        kwargs["tolerance"] = 10 * config.batch_size
        kwargs["min_size"] = config.batch_size
    replay = embodied.replay.Replay(config.batch_length, size, directory, **kwargs)
    return replay


def flatten_dict(d: DictConfig, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def override_config(old_config, new_flattened_dict, parent_key="", sep="."):
    # using regex
    for k, v in old_config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        for nk, nv in new_flattened_dict.items():
            if re.match("^" + nk + "$", new_key):
                old_config[k] = nv
        if isinstance(v, DictConfig):
            override_config(v, new_flattened_dict, new_key, sep=sep)


class Dreamerv3Agent(Agent):

    def __init__(self, cfg):
        super().__init__(cfg=cfg)

        self.env = utils.make_env(cfg, 0)

        # overwrite by humanoid_benchmark
        flattened_new_dict = flatten_dict(cfg.agent.humanoid_benchmark)
        override_config(cfg.agent, flattened_new_dict)

        eval_config = cfg.copy()
        if "humanoid" in cfg.env:
            eval_config.agent.env.humanoid.is_eval = True

        self.make_train_env = bind(utils.make_env, cfg)
        self.make_eval_env = bind(
            utils.make_env, eval_config
        )  # Avoid using this. This will render at each step so it is very slow.

        args = cfg.agent.run
        self.args = args

        self.batch_size = cfg.agent.batch_size
        self.batch_length = cfg.agent.batch_length

        logdir = embodied.Path(self.work_dir / "models" / cfg.run_name)
        logdir.mkdirs()
        self.logdir = logdir
        print("Logdir", logdir)

        # below is copied from original train_eval.py
        self.agent = make_agent(cfg)
        self.train_replay = make_replay(cfg.agent, "replay")
        self.eval_replay = make_replay(cfg.agent, "eval_replay", is_eval=True)
        self.logger = make_logger(cfg.agent, cfg.env, cfg.log_wandb, logdir)

        self.obs_buffer = ObservationBuffer(capacity=cfg.obs_buffer_capacity)

        self.step = self.logger.step
        self.usage = embodied.Usage(**args.usage)
        self.agg = embodied.Agg()
        train_episodes = defaultdict(embodied.Agg)
        self.train_epstats = embodied.Agg()
        eval_episodes = defaultdict(embodied.Agg)
        self.eval_epstats = embodied.Agg()

        batch_steps = self.batch_size * self.batch_length
        self.should_expl = embodied.when.Until(args.expl_until)
        self.should_train = embodied.when.Ratio(args.train_ratio / batch_steps)
        self.should_log = embodied.when.Clock(args.log_every)
        self.should_save = embodied.when.Clock(args.save_every)
        self.should_eval = embodied.when.Every(args.eval_every, args.eval_initial)

        @embodied.timer.section("log_step")
        def log_step(tran, worker, mode):
            episodes = dict(train=train_episodes, eval=eval_episodes)[mode]
            epstats = dict(train=self.train_epstats, eval=self.eval_epstats)[mode]

            episode = episodes[worker]
            episode.add("score", tran["reward"], agg="sum")
            episode.add("length", 1, agg="sum")
            episode.add("rewards", tran["reward"], agg="stack")

            if "success" in tran:
                episode.add("success", tran["success"], agg="sum")
            if "success_subtasks" in tran:
                episode.add("success_subtasks", tran["success_subtasks"], agg="max")

            if tran["is_first"]:
                episode.reset()

            if worker < args.log_video_streams:
                for key in args.log_keys_video:
                    if key in tran:
                        episode.add(f"policy_{key}", tran[key], agg="stack")

            for key, value in tran.items():
                if re.match(args.log_keys_sum, key):
                    episode.add(key, value, agg="sum")
                if re.match(args.log_keys_avg, key):
                    episode.add(key, value, agg="avg")
                if re.match(args.log_keys_max, key):
                    episode.add(key, value, agg="max")

            if tran["is_last"]:
                result = episode.result()
                self.logger.add(
                    {
                        "score": result["score"],
                        "length": result["length"] - 1,
                    },
                    prefix="episode",
                )
                self.logger.add(
                    {
                        "return": result.pop("score"),
                        "episode_length": result.pop("length") - 1,
                    },
                    prefix="results",
                )
                if worker < args.log_video_streams:
                    for key in args.log_keys_video:
                        if key in tran:
                            self.logger.add(
                                {"video": result[f"policy_{key}"]}, prefix="results"
                            )
                if "success" in result:
                    self.logger.add(
                        {"success": result.pop("success")}, prefix="results"
                    )
                if "success_subtasks" in result:
                    self.logger.add(
                        {"success_subtasks": result.pop("success_subtasks")},
                        prefix="results",
                    )
                rew = result.pop("rewards")
                result["reward_rate"] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
                epstats.add(result)

        fns = [bind(self.make_train_env, i) for i in range(args.num_envs)]
        self.train_driver = embodied.Driver(fns, args.driver_parallel)
        self.train_driver.on_step(lambda tran, _: self.step.increment())
        self.train_driver.on_step(self.train_replay.add)
        self.train_driver.on_step(
            lambda tran, _: self.obs_buffer.add(
                tran["vector"]
                if not self.cfg.use_action
                else np.concatenate([tran["vector"], tran["action"]])
            )
        )
        self.train_driver.on_step(bind(log_step, mode="train"))

        fns = [bind(self.make_eval_env, i) for i in range(args.num_envs)]
        self.eval_driver = embodied.Driver(fns, args.driver_parallel)
        self.eval_driver.on_step(self.eval_replay.add)
        self.eval_driver.on_step(bind(log_step, mode="eval"))
        self.eval_eps = args.eval_eps

        self.train_dataset = self.agent.dataset(
            embodied.Batch([self.train_replay.dataset] * self.batch_size)
        )
        self.eval_dataset = self.agent.dataset(
            embodied.Batch([self.eval_replay.dataset] * self.batch_size)
        )

        self.checkpoint = embodied.Checkpoint(logdir / "checkpoint.ckpt")
        self.checkpoint.step = self.step
        self.checkpoint.agent = self.agent
        self.checkpoint.train_replay = self.train_replay
        self.checkpoint.eval_replay = self.eval_replay
        if args.from_checkpoint:
            self.checkpoint.load(args.from_checkpoint)
        self.checkpoint.load_or_save()
        self.should_save(self.step)  # Register that we just saved.

    def get_obs_shape(self):
        return self.env.observation_space.shape[0]

    def get_action_shape(self):
        return self.env._env.action_space.shape[0]

    def prepare(self, reward_fn=None):

        carry = [self.agent.init_train(self.batch_size)]

        def train_step(tran, worker):
            if (
                len(self.train_replay) < self.batch_size
                or self.step < self.args.train_fill
            ):
                return

            for _ in range(self.should_train(self.step)):
                with embodied.timer.section("dataset_next"):
                    batch = next(self.train_dataset)

                    if reward_fn is not None:
                        _reward = batch["reward"]

                        # obs: jax -> numpy -> torch
                        obs = jax.device_get(batch["vector"])
                        obs = np.array(obs)

                        if self.cfg.use_action:
                            action = jax.device_get(batch["action"])
                            obs = np.concatenate([obs, action], axis=-1)

                        obs = torch.from_numpy(obs).cuda()

                        obs_shape = obs.shape
                        obs = obs.reshape(-1, obs.shape[-1])
                        reward = reward_fn(obs)

                        # reward: torch -> numpy -> jax
                        reward = reward.reshape(obs_shape[:-1]).cpu().numpy()
                        reward = jax.device_put(reward)

                        assert reward.shape == _reward.shape
                        batch["reward"] = reward

                outs, carry[0], mets = self.agent.train(batch, carry[0])
                self.agg.add(mets, prefix="train")

        self.train_driver.on_step(train_step)
        self.train_driver.reset(self.agent.init_policy)

        self._prepared = True

    def evaluate(self, video_prefix=None, num_eval_eps=5):
        reward_list = []
        eps_cnt = []

        progress_bar = tqdm(
            total=num_eval_eps * 1000, desc="Evaluating"
        )  # 1000 is hacky

        policy = lambda *args: self.agent.policy(*args, mode="eval")
        fns = [bind(self.make_train_env, i) for i in range(self.args.num_envs)]
        evaluate_driver = embodied.Driver(fns, self.args.driver_parallel)

        def add_reward(tran, worker=0):
            reward_list.append(tran["reward"])
            if tran["is_last"]:
                eps_cnt.append(1)

        evaluate_driver.on_step(add_reward)
        evaluate_driver.on_step(lambda *args: progress_bar.update(1))

        evaluate_driver.reset(self.agent.init_policy)
        evaluate_driver(policy, episodes=num_eval_eps)

        progress_bar.close()

        return np.sum(reward_list) / len(eps_cnt)

    def my_evaluate(self, actor):
        num_eval_eps = self.cfg.num_eval_episodes
        reward_list = []
        for _ in range(num_eval_eps):
            episode_reward = 0
            obs = self.env.step(action={"action": 0, "reset": True})["vector"]
            done = False
            length = 0
            while not done:
                action = actor(torch.tensor(obs).float().cuda()).mean
                action = action.cpu().detach().numpy()
                # action = self.env._env.action_space.sample()
                timestep = self.env.step(action={"action": action, "reset": False})
                obs = timestep["vector"]
                reward = timestep["reward"]
                done = timestep["is_last"]
                episode_reward += reward
                length += 1
            print("Episode reward:", episode_reward, "Episode length:", length)
            reward_list.append(episode_reward)
        print("episode reward list", reward_list)
        return np.mean(reward_list)

    def run_bc(self, dataset, num_train_epochs):
        from agent.sac.actor import DiagGaussianActor

        actor = DiagGaussianActor(
            obs_dim=self.get_obs_shape(),
            action_dim=19,
            hidden_dim=self.cfg.agent.actor.units,
            hidden_depth=self.cfg.agent.actor.layers - 1,
            log_std_bounds=[
                np.log(self.cfg.agent.actor.minstd),
                np.log(self.cfg.agent.actor.maxstd),
            ],
        ).to(self.device)
        actor_optimizer = torch.optim.Adam(
            actor.parameters(), lr=self.cfg.agent.actor_opt.lr
        )

        for i in range(num_train_epochs):

            metrics = dict()

            if i % 2000 == 0:
                ret = self.my_evaluate(actor=actor)
                metrics["episode_reward"] = ret
                print(f"eval: {ret}")

            obs = torch.tensor(dataset[:, : self.get_obs_shape()]).to(self.device)
            action = torch.tensor(dataset[:, self.get_obs_shape() :]).to(self.device)

            dist = actor(obs)
            if self.cfg.alg.loss == "mle":
                # clip action
                action = torch.clamp(action, -1 + 1e-5, 1 - 1e-5)
                log_prob = dist.log_prob(action).sum(-1, keepdim=True)
                loss = -log_prob.mean()
            elif self.cfg.alg.loss == "mse":
                loss = F.mse_loss(dist.mean, action)
            else:
                raise ValueError(f"loss_type {self.cfg.alg.loss} not supported")
            metrics["loss"] = loss.item()
            actor_optimizer.zero_grad()
            loss.backward()
            actor_optimizer.step()

            if self.cfg.log_wandb:
                wandb.log(metrics)

            if i % 100 == 0:
                print(f"epoch {i}, loss: {metrics['loss']}")

    def run(self, num_train_steps):
        assert self._prepared, "Call prepare() before run()"
        assert (
            num_train_steps > 1000
        ), "Training for less than 1000 steps is not supported"

        train_policy = lambda *args: self.agent.policy(
            *args, mode="explore" if self.should_expl(self.step) else "train"
        )
        eval_policy = lambda *args: self.agent.policy(*args, mode="eval")

        end_step = self.step + num_train_steps
        init_step = self.step

        while self.step < end_step:
            if self.should_eval(self.step):
                print("Start evaluation")
                self.eval_driver.reset(self.agent.init_policy)
                self.eval_driver(eval_policy, episodes=self.eval_eps)
                self.logger.add(self.eval_epstats.result(), prefix="epstats")
                if len(self.eval_replay):
                    self.logger.add(
                        self.agent.report(next(self.eval_dataset)), prefix="eval"
                    )
                print("Done evaluation")

            self.train_driver(train_policy, steps=10)

            if self.should_log(self.step):
                self.logger.add(self.agg.result())
                self.logger.add(self.train_epstats.result(), prefix="epstats")
                if len(self.train_replay):
                    self.logger.add(
                        self.agent.report(next(self.train_dataset)), prefix="report"
                    )
                self.logger.add(embodied.timer.stats(), prefix="timer")
                self.logger.add(self.train_replay.stats(), prefix="replay")
                self.logger.add(self.usage.stats(), prefix="usage")
                self.logger.write(fps=True)

            if self.should_save(self.step):
                self.checkpoint.save(
                    filename=self.logdir / int(self.step) / "checkpoint.ckpt"
                )

    def collect_obs(self, num_obs, save_action, verbose=False):

        obs_list = []

        progress_bar = tqdm(total=num_obs, desc="Collecting obs")

        policy = lambda *args: self.agent.policy(*args, mode="eval")
        fns = [bind(self.make_train_env, i) for i in range(self.args.num_envs)]
        collect_driver = embodied.Driver(fns, self.args.driver_parallel)

        def add_obs(tran, worker=0):
            obs_list.append(
                tran["vector"]
                if not save_action
                else np.concatenate([tran["vector"], tran["action"]])
            )

        collect_driver.on_step(add_obs)
        collect_driver.on_step(lambda *args: progress_bar.update(1))
        collect_driver.reset(self.agent.init_policy)
        collect_driver(policy, steps=num_obs)

        progress_bar.close()

        return obs_list[:num_obs], -1

    def load_agent(self, path):
        self.checkpoint.load(path / "checkpoint.ckpt")
