import wandb
import hydra
from omegaconf import OmegaConf

import utils


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg):
    if cfg.log_wandb:
        wandb.init(
            project="expert",
            config=OmegaConf.to_container(cfg),
            name=cfg.run_name,
            monitor_gym=True,
            save_code=True,
        )
    utils.set_seed_everywhere(cfg.seed)
    agent = utils.make_agent(cfg)
    agent.prepare()
    agent.run(num_train_steps=cfg.expert_num_train_steps)


if __name__ == "__main__":
    main()
