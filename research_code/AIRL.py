from feature_extractors import VQVAEFeatureExtractor
import torch
import torch.nn as nn
import numpy as np
import einops
import minerl
import os
#from imitation.rewards.reward_nets import RewardNet
import argparse
import stable_baselines3 as sb3
from datasets import AIRLDataset
from imitation.util import logger, util
from imitation.algorithms import adversarial

from gym import ObservationWrapper
from gym.spaces import Box

from vqvae import VQVAE

from functools import partial

def helper(env, i, visual_model_path):
    return LatentObservationWrapper(env, visual_model_path)

class LatentObservationWrapper(ObservationWrapper):
    def __init__(self, env, visual_model_path):
        super().__init__(env)
        
        # modify obersvation space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2048,))
        
        # load visual model
        self.model = VQVAE.load_from_checkpoint(visual_model_path)
        self.model.eval()
        
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=self.model.hparams.embedding_dim, out_channels=256, kernel_size=3, padding=1, stride=2), # 16 -> 8
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2), # 8 -> 4
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2), # 4 -> 2
            nn.GELU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=2), # 2 -> 1
        )
        
    def observation(self, obs):
        # modify obs
        
        # prepare image for conv nets
        pov = torch.from_numpy(obs['pov'].astype(np.float32)) / 255
        pov = einops.rearrange(pov, 'h w c -> c h w')[None]
        
        # apply pretrained vqvae
        z_q, *_ = self.model.encode_only(pov)
        #z_q.shape = [1, 32, 16, 16]
        
        # apply conv net -> this should train online (?) #TODO: Check that this actually trains
        encoded_pov = einops.rearrange(self.conv_net(z_q), 'b c h w -> (b c h w)')
        # [2048]
        
        obs['pov'] = encoded_pov
        return obs

def main(
    env_name, 
    data_dir, 
    log_dir, 
    expert_batch_size, 
    n_envs, 
    total_timesteps, 
    gen_algo_steps, 
    disable_verbose,
    discount_factor,
    visual_model_path
):
    
    # set up data loading
    data = AIRLDataset(env_name, data_dir, expert_batch_size, num_epochs=1)
    dataloader = torch.utils.data.DataLoader(data)

    wrapper_cls = partial(helper, visual_model_path=visual_model_path)

    # set up vectorized env
    venv = util.make_vec_env(env_name, n_envs=n_envs, post_wrappers=[wrapper_cls])

    # set up logger
    logger.configure(os.path.join(log_dir, 'AIRL/'))
    
    # define reward kwargs
    reward_net_kwargs = {
        'reward_hid_sizes':[128,128],
        'potential_hid_sizes':[128,128],
        'use_next_state':True,
        'discount_factor':discount_factor
    }
    
    # init airl trainer
    airl_trainer = adversarial.AIRL(
        venv,
        expert_data=dataloader,
        expert_batch_size=expert_batch_size,
        #reward_net_cls=None,
        reward_net_kwargs=reward_net_kwargs,
        #discrim_kwargs=None,
        gen_algo=sb3.PPO('MlpPolicy', venv, verbose=int(~disable_verbose), n_steps=gen_algo_steps),
        custom_logger=logger
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
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--visual_model_path', type=str, default=None)
    parser.add_argument('--disable_verbose', action='store_false')
    
    args = parser.parse_args()
    
    main(**vars(args))