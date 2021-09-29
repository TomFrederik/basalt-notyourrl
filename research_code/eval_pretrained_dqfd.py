import argparse
import torch
import gym
import os
import random
import einops 
import numpy as np
import cv2

from DQfD_models import QNetwork
from DQfD_utils import preprocess_non_pov_obs
from action_shaping import ActionWrapper

def main(
    env_name,
    log_dir,
    num_episodes,
    max_episode_len,
    model_id,
    video_dir,
    epsilon
):
    # set run id
    video_dir = os.path.join(video_dir, env_name, model_id)

    # check that video_dir exists
    os.makedirs(video_dir, exist_ok=True)

    # load model
    model_path = os.path.join(log_dir, env_name, model_id, 'model.pt')
    q_net: QNetwork = torch.load(model_path)
    q_net.eval()

    # init env
    env = gym.make(env_name)
    env = ActionWrapper(env, env_name)

    for i in range(num_episodes):
        print(f'\nStarting episode {i+1}')
        # reset env
        obs = env.reset()
        done = False
        steps = 0
        pov_list = []

        while not done:
            print(f'{steps = }') # for debugging
            # save pov obs for video creation
            pov_list.append(obs['pov'])

            # extract pov and inv from obs and convert to torch tensors
            pov = einops.rearrange(obs['pov'], 'h w c -> 1 c h w').astype(np.float32) / 255
            inv = einops.rearrange(preprocess_non_pov_obs(obs), 'd -> 1 d')
            pov = torch.from_numpy(np.array(pov))
            inv = torch.from_numpy(np.array(inv))

            # compute q_values
            q_values = q_net.forward(dict(pov=pov, inv=inv))

            # select action
            if random.random() < epsilon:
                action = random.randint(q_net.num_actions)
            else:
                action = torch.argmax(q_values).squeeze().item()

            # take action in environment
            obs, rew, done, info = env.step(action)

            # check stopping criterion
            steps += 1
            if steps >= max_episode_len and (not done):
                print(f'Stopping prematurely after {max_episode_len} steps')
                break
        
        print(f'\nFinished episode {i+1}. Creating video..\n')
        out = cv2.VideoWriter(os.path.join(video_dir, f'{i}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'), 25, (64,64))

        for i in range(len(pov_list)):
            out.write(pov_list[i])
        out.release()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLBasaltFindCave-v0')
    parser.add_argument('--log_dir', type=str, default='/home/lieberummaas/datadisk/basalt-notyourrl/run_logs/')
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--max_episode_len', type=int, default=2000)
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--video_dir', type=str, default='/home/lieberummaas/datadisk/basalt-notyourrl/pretrain_videos')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Epsilon for epsilon-greedy behaviour')

    main(**vars(parser.parse_args()))
