import hydra
from pathlib import Path

import utils


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg):

    utils.set_seed_everywhere(cfg.seed)
    agents = [utils.make_agent(cfg)]

    agent_model_paths = utils.get_agent_model_paths(cfg)
    for agent, path in zip(agents, agent_model_paths):
        agent.load_agent(path)

    # evaluate it first
    avg_episode_return = agent.evaluate(video_prefix="EVAL")
    print(f"avg episode return: {avg_episode_return}", flush=True)

    # collect observations --- training set and validation set
    obs_save_dir = Path.cwd() / "models" / (cfg.expert_path)
    obs_save_dir.mkdir(parents=True, exist_ok=True)
    retain_every = cfg.expert_dataset_retain_every
    assert retain_every >= 1

    train_obs_batch = utils.load_or_collect_obs(
        obs_save_dir
        / f"train_obs_{cfg.expert_dataset_num_obs}{'_a' if cfg.use_action else ''}.npy",
        agents=agents,
        num_obs=cfg.expert_dataset_num_obs,
        retain_every=retain_every,
        use_action=cfg.use_action,
    )

    val_obs_batch = utils.load_or_collect_obs(
        obs_save_dir
        / f"val_obs_{cfg.expert_dataset_num_obs}{'_a' if cfg.use_action else ''}.npy",
        agents=agents,
        num_obs=cfg.expert_dataset_num_obs,
        retain_every=retain_every,
        use_action=cfg.use_action,
    )

    print(
        f"training set num obs: {train_obs_batch.shape[0]}\nvalidation set num obs: {val_obs_batch.shape[0]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
