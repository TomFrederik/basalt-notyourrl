import pickle
import uuid
from pathlib import Path

import cv2
import einops
import gym
import numpy as np
import time
import torch
from stable_baselines3 import A2C, DQN

from reward_model import RewardModel


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

        # Overwrite next_state to use image
        img = self.env.render(mode="rgb_array")
        # Resize to the dimensions we expect later
        img = cv2.resize(img, dsize=(64, 64))

        img = torch.as_tensor(img, dtype=torch.float32)
        img = einops.rearrange(img, 'h w c -> 1 c h w') / 255
        rewards = self.model(img)
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

    TRAIN_STEPS = 10000
    NUM_TRAJECTORIES = 20
    OUT_DIR = Path("./trained_output_trajectories")
    GYM_ENV = 'CartPole-v1'
    REWARD_MODEL_PATH = "/home/junshern.chan/git/basalt-notyourrl/rlhpd/DRLfHP/models/iconic-darkness-43/00556.pt"
    POLICY_MODELS_DIR = Path("./policy_models")
    run_name = time.strftime('%Y%m%d-%H%M%S')
    POLICY_PATH = "/home/junshern.chan/git/basalt-notyourrl/rlhpd/DRLfHP/policy_models/20210925-233103_cc35ef32-aced-41d1-9b9c-7e53e8f34aa0.zip"

    env = gym.make(GYM_ENV)
    env = RewardModelWrapper(env, REWARD_MODEL_PATH)

    if POLICY_PATH is None:
        print("Training policy")
        policy_model = A2C('MlpPolicy', env, verbose=1)
        policy_model.learn(total_timesteps=TRAIN_STEPS)
        policy_model_path = POLICY_MODELS_DIR / run_name
        policy_model.save(policy_model_path)
    else:
        policy_model_path = POLICY_PATH
        print("Loading policy from", policy_model_path)

    print("Generating trajectories")
    generate_trajectories(policy_model_path, env, NUM_TRAJECTORIES, OUT_DIR / run_name)
