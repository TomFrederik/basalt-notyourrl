import sys
sys.path.append("../..")
import pickle
import time
import uuid
from pathlib import Path

import cv2
import einops
import gym
import minerl
import numpy as np
import torch
import torch.nn as nn
import wandb
from gym import spaces
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wandb.integration.sb3 import WandbCallback

import common.utils as utils
from common.action_shaping import ActionWrapper
from DRLfHP.reward_model import RewardModel


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == "img":
                # We will just downsample one channel of the image by 4x4 and flatten.
                # Assume the image is single-channel (subspace.shape[0] == 0)
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "vec":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)

class ObservationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.observation_space = spaces.Dict({
            "pov": spaces.Box(low=0, high=255, shape=(64, 64, 3)),
            "compassAngle": spaces.Box(low=-180.0, high=180.0, shape=()),
            })

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        # Preprocess state
        img = np.array(next_state['pov'])
        assert img.shape == (64, 64, 3), img.shape
        img = einops.rearrange(img, 'h w c -> 1 c h w') / 255
        vec = np.array([next_state['compass']['angle']])
        assert vec.shape == (1,), vec.shape
        vec = einops.rearrange(vec, '1 -> 1 1')
        next_state = {
            'pov': img,
            'compassAngle': vec,
        }

        return next_state, reward, done, info

class RewardModelWrapper(gym.Wrapper):
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
        rewards = self.model(
            torch.as_tensor(next_state['img'], dtype=torch.float32), 
            torch.as_tensor(next_state['vec'], dtype=torch.float32)
            )
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

if __name__ == "__main__":
    wandb_project_name = "policy-navigate"
    # wandb.init(project=wandb_project_name, entity='junshern')
    wandb.init(project=wandb_project_name, entity='junshern', mode="disabled")

    # save_dir = Path(options.save_root_dir) / wandb.run.name
    # save_dir.mkdir(parents=True, exist_ok=True)

    cfg = wandb.config
    cfg.rand_seed = 1

    utils.set_seeds(cfg.rand_seed)

    cfg.train_steps = 10000
    cfg.num_trajectories = 20
    cfg.out_dir = Path("./navigate_densereward_trained_output_trajectories")
    cfg.gym_env = 'MineRLNavigateDense-v0' # 'CartPole-v1'
    # reward_model_path = "/home/junshern.chan/git/basalt-notyourrl/rlhpd/DRLfHP/models/iconic-darkness-43/00556.pt"
    cfg.reward_model_path = None # "/home/junshern.chan/git/basalt-notyourrl/rlhpd/DRLfHP/navigate_models/gallant-water-2/090900.pt"
    cfg.policy_models_dir = Path("./navigate_policy_models")
    cfg.run_name = time.strftime('%Y%m%d-%H%M%S')
    cfg.policy_path = None # "/home/junshern.chan/git/basalt-notyourrl/rlhpd/DRLfHP/policy_models/20210925-233103_cc35ef32-aced-41d1-9b9c-7e53e8f34aa0.zip"

    env = gym.make(cfg.gym_env)
    env = ActionWrapper(env, 'MineRLBasaltFindCave-v0')
    env = ObservationWrapper(env)
    if cfg.reward_model_path is not None:
        env = RewardModelWrapper(env, cfg.reward_model_path)

    print(env.action_space)
    print(env.observation_space)
    if cfg.policy_path is None:
        # obs = env.reset()
        # done = False
        # while not done:
        #     # action, _state = policy_model.predict(obs, deterministic=True)
        #     action = env.action_space.sample()
        #     obs, reward, done, info = env.step(action)
        #     env.render()

        print("Training policy")
        # policy_kwargs = dict(
        #     features_extractor_class=CustomCombinedExtractor,
        #     features_extractor_kwargs=dict(features_dim=128),
        # )
        policy_model = A2C('MultiInputPolicy', env, verbose=1)
        policy_model.learn(total_timesteps=cfg.train_steps, callback=WandbCallback)
        # policy_model_path = policy_models_dir / run_name
        # policy_model.save(policy_model_path)
    # else:
    #     policy_model_path = policy_path
    #     print("Loading policy from", policy_model_path)

    # print("Generating trajectories")
    # generate_trajectories(policy_model_path, env, num_trajectories, out_dir / run_name)
