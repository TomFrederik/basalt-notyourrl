import argparse
import random
import os

import torch
import gym
import einops 
import numpy as np
import cv2

import minerl
from common.DQfD_models import QNetwork
from common.state_shaping import preprocess_non_pov_obs, StateWrapper
from common.DQfD_utils import RewardActionWrapper, DummyRewardModel


def main(
    env_name,
    num_episodes,
    max_episode_len,
    model_path,
    video_dir,
    epsilon
):
    # check that video_dir exists
    os.makedirs(video_dir, exist_ok=True)

    # load model
    q_net: QNetwork = torch.load(model_path)
    q_net.eval()

    # init env
    env = gym.make(env_name)
    env = StateWrapper(env)
    env = RewardActionWrapper(env_name, env, DummyRewardModel())

    for i in range(num_episodes):
        print(f'\nStarting episode {i+1}')
        # reset env
        obs = env.reset()
        done = False
        steps = 0
        pov_list = []

        while not done:
            # save pov obs for video creation
            pov_list.append((einops.rearrange(obs['pov'], 'c h w -> h w c') * 255).astype(np.uint8))

            # extract pov and inv from obs and convert to torch tensors
            pov = torch.from_numpy(obs['pov'][None])
            vec = torch.from_numpy(obs['vec'][None])

            # compute q_values
            q_values = q_net.forward(pov, vec)

            # select action
            if random.random() < epsilon:
                action = random.randint(0, q_net.num_actions)
            else:
                action = torch.argmax(q_values).squeeze().item()

            # take action in environment
            obs, rew, done, info = env.step(action)

            # check stopping criterion
            steps += 1
            if steps >= max_episode_len and (not done):
                print(f'Stopping prematurely')
                break
        
        print(f'\nFinished episode {i+1} after {steps} steps. Creating video..\n')
        out = cv2.VideoWriter(os.path.join(video_dir, f'{i}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 25, (64,64))

        for i in range(len(pov_list)):
            out.write(pov_list[i][:,:,::-1])
        out.release()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLBasaltFindCave-v0')
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--max_episode_len', type=int, default=2000)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--video_dir', type=str, default='./output/videos')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Epsilon for epsilon-greedy behaviour')

    main(**vars(parser.parse_args()))
