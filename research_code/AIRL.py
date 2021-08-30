from feature_extractors import VQVAEFeatureExtractor
import torch
import torch.nn as nn
import minerl
import os
#from imitation.rewards.reward_nets import RewardNet
import argparse
import stable_baselines3 as sb3
from datasets import AIRLDataset
from imitation.util import logger, util
from imitation.algorithms import adversarial

def main(env_name, data_dir, log_dir, expert_batch_size, n_envs, total_timesteps, gen_algo_steps, disable_verbose):
    
    # set up data loading
    data = AIRLDataset(env_name, data_dir, expert_batch_size, num_epochs=1)
    dataloader = torch.utils.data.DataLoader(data)

    # set up vectorized env
    venv = util.make_vec_env(env_name, n_envs=n_envs)

    # set up logger
    logger.configure(os.path.join(log_dir, 'AIRL/'))
    
    # init airl trainer
    airl_trainer = adversarial.AIRL(
        venv,
        expert_data=dataloader,
        expert_batch_size=expert_batch_size,
        gen_algo=sb3.PPO('MlpPolicy', venv, verbose=int(~disable_verbose), n_steps=gen_algo_steps)
    )
    
    # train
    airl_trainer.train(total_timesteps=total_timesteps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./run_logs')
    parser.add_argument('--expert_batch_size', type=int, default=512)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=2048)
    parser.add_argument('--gen_algo_steps', type=int, default=1024)
    parser.add_argument('--disable_verbose', action='store_false')
    
    args = parser.parse_args()
    
    main(**vars(args))