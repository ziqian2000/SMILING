import hydra
import utils

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg):

    utils.set_seed_everywhere(cfg.seed)
    agents = [utils.make_agent(cfg)]

    assert cfg.alg.name == "smiling", "Please turn on the smiling algorithm"

    agent_model_paths = utils.get_agent_model_paths(cfg)
    for agent, path in zip(agents, agent_model_paths):
        agent.load_agent(path)

    avg_episode_return = agents[0].evaluate("EVAL")
    print(f"avg episode return: {avg_episode_return}", flush=True)


if __name__ == "__main__":
    main()
