import argparse
import pickle
from pathlib import Path

import common.utils as utils
import cv2
import einops
import gym
import minerl
import torch
import wandb
from common.reward_model import RewardModel
from gym import spaces
from stable_baselines3 import A2C
from wandb.integration.sb3 import WandbCallback


class ActionWrapper(gym.ActionWrapper):
    """
    MineRL expects the action to be presented in a dictionary with key 'vector'.
    The model instead returns the action as a plain vector, so we simply wrap
    the vector in a dictionary as required by the environment.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = spaces.Box(low=-1.0499999523162842, high=1.0499999523162842, shape=(64,))
    
    def action(self, act):
        return {
            'vector': act,
        }

class RewardModelWrapper(gym.Wrapper):
    """
    Replace the basic reward signal with our own learned reward function.
    """
    def __init__(self, env, reward_model_path):
        super().__init__(env)
        self.env = env
        self.load_model(reward_model_path)

    def load_model(self, model_path):
        self.model = RewardModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # Predict reward with reward model
        pov = torch.as_tensor(next_state['pov'], dtype=torch.float32)
        pov = einops.rearrange(pov, 'h w c -> 1 c h w')
        vec = torch.as_tensor(next_state['vector'], dtype=torch.float32)
        vec = einops.rearrange(vec, 'v -> 1 v')
        rewards = self.model(pov, vec)
        assert rewards.shape == (1, 1), rewards.shape
        reward = rewards.squeeze().item()

        return next_state, reward, done, info


def generate_trajectories(policy_model_path, env, num_trajectories, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_model = A2C.load(policy_model_path, env=env)

    for i in range(num_trajectories):
        print("Generating trajectory", i)
        
        obs = env.reset()
        trajectory = []

        done = False
        while not done:
            action, _state = policy_model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Overwrite next_state to use image
            img = env.render(mode="rgb_array")
            # Resize to the dimensions we expect later
            img = cv2.resize(img, dsize=(64, 64))

            trajectory.append((img, action, reward))

        # Save pickle
        outfile = out_dir / Path(f"{i:05d}").with_suffix('.pickle')
        with open(outfile, 'wb') as f:
            pickle.dump(trajectory, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    # Params
    cfg = utils.load_config(options.config_file)

    wandb.init(project=cfg.policy.wandb_project, entity=cfg.wandb_entity)
    # wandb.init(project=cfg.policy.wandb_project, entity=cfg.wandb_entity, mode="disabled")

    save_dir = Path(cfg.policy.save_dir) / wandb.run.name
    save_dir.mkdir(parents=True, exist_ok=True)

    utils.set_seeds(cfg.policy.rand_seed)

    env = gym.make(cfg.env_task)
    env = ActionWrapper(env)
    # if cfg.policy.reward_model_path is not None:
    #     env = RewardModelWrapper(env, cfg.policy.reward_model_path)

    if cfg.policy.policy_path is None:
        policy_model = A2C('MultiInputPolicy', env, verbose=1)
        print("Training policy")
        policy_model.learn(total_timesteps=cfg.policy.train_steps, callback=WandbCallback())
        policy_model_path = save_dir / "policy"
        policy_model.save(policy_model_path)
    else:
        # Skip training if the user has already specified a policy
        policy_model_path = cfg.policy.policy_path
        print("Skipping training. Loading policy from", policy_model_path)

    print("Generating trajectories")
    num_traj = cfg.policy.num_trajectories
    out_dir = Path(cfg.policy.out_dir) / wandb.run.name
    generate_trajectories(policy_model_path, env, num_traj, out_dir)
