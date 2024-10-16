import numpy as np
import torch
from torch import nn
import os
import random
import importlib
import os
from pathlib import Path

import dmc2gym
from algorithm.diffusion import Diff
from algorithm.dac import DAC


def get_agent_model_paths(cfg):
    expert_paths = [cfg.expert_path]
    paths = []

    for expert_path in expert_paths:
        if cfg.agent.name == "sac":
            paths.append(Path.cwd() / "models" / expert_path)
        elif cfg.agent.name == "dreamerv3":
            paths.append(Path.cwd() / "models" / expert_path)
        else:
            raise ValueError(f"Unknown agent {expert_path}")

    return paths


def make_alg(cfg, obs_dim, action_dim):
    print(f"Making algorithm: {cfg.alg.name}")
    if cfg.alg.name == "smiling":
        return Diff(
            cfg,
            obs_dim=obs_dim if not cfg.use_action else obs_dim + action_dim,
        )
    elif cfg.alg.name == "dac":
        return DAC(
            cfg,
            obs_dim=obs_dim if not cfg.use_action else obs_dim + action_dim,
        )
    elif cfg.alg.name == "bc":
        return None
    else:
        raise NotImplementedError


def make_agent(cfg):
    print(f"Making agent: {cfg.agent.name}")
    if cfg.agent.name == "sac":
        import agent.sac.agent

        return agent.sac.agent.SACAgent(cfg)
    elif cfg.agent.name == "dreamerv3":
        import agent.dreamerv3.agent

        return agent.dreamerv3.agent.Dreamerv3Agent(cfg)
    else:
        raise NotImplementedError


def make_env(cfg, index=0, **overrides):

    suite, task = cfg.env.split("_", 1)

    if suite != "humanoidbench":  # dmc

        """Helper function to create dm_control environment"""
        if suite == "ball":
            domain_name = "ball_in_cup"
            task_name = "catch"
        else:
            domain_name = cfg.env.split("_")[0]
            task_name = "_".join(cfg.env.split("_")[1:])

        env = dmc2gym.make(
            domain_name=domain_name,
            task_name=task_name,
            seed=cfg.seed,
            visualize_reward=True,
        )
        env.seed(cfg.seed)
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1

        return env

    else:  # humanoidbench
        ctor = {
            "dummy": "embodied.envs.dummy:Dummy",
            "gym": "embodied.envs.from_gym:FromGym",
            "dm": "embodied.envs.from_dmenv:FromDM",
            "crafter": "embodied.envs.crafter:Crafter",
            "dmc": "embodied.envs.dmc:DMC",
            "atari": "embodied.envs.atari:Atari",
            "atari100k": "embodied.envs.atari:Atari",
            "dmlab": "embodied.envs.dmlab:DMLab",
            "minecraft": "embodied.envs.minecraft:Minecraft",
            "loconav": "embodied.envs.loconav:LocoNav",
            "pinpad": "embodied.envs.pinpad:PinPad",
            "langroom": "embodied.envs.langroom:LangRoom",
            "humanoidbench": "embodied.envs.from_gymnasium:FromGymnasium",
        }[suite]
        if isinstance(ctor, str):
            module, cls = ctor.split(":")
            module = importlib.import_module("agent.dreamerv3." + module)
            ctor = getattr(module, cls)
        if suite == "humanoidbench":
            suite = "humanoid"
        kwargs = cfg.agent.env.get(suite, {})
        kwargs.update(overrides)
        if kwargs.get("use_seed", False):
            kwargs["seed"] = hash((cfg.seed, index))
        env = ctor(task, **kwargs)
        return wrap_env(env, cfg)


def wrap_env(env, config):
    from agent.dreamerv3.embodied import wrappers

    """this is only used for humanoidbench envs"""
    args = config.agent.wrapper
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif not space.discrete:
            env = wrappers.NormalizeAction(env, name)
            if args.discretize:
                env = wrappers.DiscretizeAction(env, name, args.discretize)
    env = wrappers.ExpandScalars(env)
    if args.length:
        env = wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = wrappers.ClipAction(env, name)
    return env


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(*path_parts):
    dir_path = os.path.join(*path_parts)
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None
    ):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod)
        self.apply(weight_init)

    def forward(self, x):
        return self.trunk(x)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()


def load_or_collect_obs(path, agents, num_obs, retain_every, use_action: bool):
    if path.exists():
        obs_batch = np.load(path)
        print("obs loaded from", path, flush=True)
        print("obs shape:", obs_batch.shape, flush=True)
    else:
        assert (
            num_obs % len(agents) == 0
        ), "num_obs must be divisible by the number of agents"

        num_obs_per_agent = num_obs // len(agents)
        obs_list = []
        for agent in agents:
            agent_obs_list, episode = agent.collect_obs(
                num_obs_per_agent, save_action=use_action, verbose=True
            )
            obs_list.extend(agent_obs_list)
        obs_batch = np.stack(obs_list, dtype=np.float32)
        assert obs_batch.shape[0] == num_obs
        np.save(path, obs_batch)
        print("obs saved at", path, flush=True)
        print("obs shape:", obs_batch.shape, flush=True)

    obs_batch = obs_batch[::retain_every]

    return obs_batch
