import os
import hydra
import imageio

import utils


class VideoRecorder(object):
    def __init__(self, root_dir, height=256, width=256, camera_id=0, fps=30):
        self.save_dir = utils.make_dir(root_dir, "video") if root_dir else None
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled

    def record(self, env):
        if self.enabled:
            frame = env.render(
                mode="rgb_array",
                height=self.height,
                width=self.width,
                camera_id=self.camera_id,
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


@hydra.main(version_base="1.2", config_path="./config", config_name="train")
def main(cfg):
    cfg.save_video = True

    # utils.set_seed_everywhere(cfg.seed)
    agents = [utils.make_agent(cfg)]

    agent_model_paths = utils.get_agent_model_paths(cfg)
    for agent, path in zip(agents, agent_model_paths):
        agent.load_agent(path)

    for i, agent in enumerate(agents):
        print(f"===== Making video for agent {i} =====", flush=True)
        avg_return = agents[i].evaluate(video_prefix=f"agent{i}")
        print(f"avg episode return: {avg_return}", flush=True)


if __name__ == "__main__":
    main()
