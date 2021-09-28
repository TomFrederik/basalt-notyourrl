"""
TODO

generate_clips.py
Run the policy and collect trajectories (incl rewards) -> Save those to file as .pkl
For N pairs:
    Select a pair of trajectories
    simulate judgement -> (clip1, clip2, judgement)
output: A folder of clips (.pkl containing list of (S,A,R,S,D) tuples)

"""

import cv2
import gym
import numpy as np
import pickle
import uuid
from pathlib import Path
from stable_baselines3 import A2C, DQN

TOTAL_TRAIN_STEPS = 10000
TRAIN_STAGES = 10
CLIPS_PER_STAGE = 10
EVAL_EPISODE_STEPS = 100
CLIP_LENGTH = 30 # Num frames (=steps)
OUT_DIR = Path("./output")
GYM_ENV = 'CartPole-v1'

env = gym.make(GYM_ENV)
model = A2C('MlpPolicy', env, verbose=1)

OUT_DIR.mkdir(parents=True, exist_ok=True)

for k in range(TRAIN_STAGES):
    print(k)
    model.learn(total_timesteps=TOTAL_TRAIN_STEPS / TRAIN_STAGES)
    obs = env.reset()
    saved_clips = 0
    current_clip = []
    # for i in range(EVAL_EPISODE_STEPS):
    while saved_clips < CLIPS_PER_STAGE:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if GYM_ENV == 'CartPole-v1':
            # Override reward in CartPole which does not give instantaneous feedback
            # Source: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
            # Observation:
            # Type: Box(4)
            # Num     Observation               Min                     Max
            # 0       Cart Position             -4.8                    4.8
            # 1       Cart Velocity             -Inf                    Inf
            # 2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
            # 3       Pole Angular Velocity     -Inf                    Inf
            pos, vel, angle, angular_vel = obs
            # reward = np.cos(angle * np.deg2rad(90) / np.deg2rad(24)) - vel
            reward = - (np.abs(vel) + np.abs(angular_vel)) # Minimize any motion
        img = env.render(mode="rgb_array")
        img = cv2.resize(img, dsize=(64, 64)) # Resize to the dimensions we expect later

        current_clip.append((img, action, reward))
        if len(current_clip) == CLIP_LENGTH:
            # Save pickle
            unique_filename = str(uuid.uuid4())
            outfile = (OUT_DIR / unique_filename).with_suffix('.pickle')
            with open(outfile, 'wb') as f:
                pickle.dump(current_clip, f)
            saved_clips += 1
            current_clip = [] # Empty the clip before start the next one
        if done:
            obs = env.reset()
            current_clip = []