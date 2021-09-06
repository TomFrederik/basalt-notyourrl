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
from stable_baselines3.common.vec_env import VecEnv
from datasets import AIRLDataset
from imitation.util import logger, util
from imitation.algorithms import adversarial

from gym import ObservationWrapper, ActionWrapper
from gym.spaces import Box


from vqvae import VQVAE

from functools import partial

def helper(env, i, wrapper_cls, wrapper_kwargs):
    return wrapper_cls(env, **wrapper_kwargs)

class ObservationPOVWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # modify obersvation space
        self.observation_space = Box(low=0, high=1, shape=(3,64,64,))
        
    def observation(self, obs):        
        # prepare image for conv nets
        pov = torch.from_numpy(obs['pov'].astype(np.float32)) / 255
        pov = einops.rearrange(pov, 'h w c -> c h w')
        return pov

class ReversibleActionWrapper(ActionWrapper):
    def wrap_action(self, inner_action):
        """
        :param inner_action: An action in the format of the innermost env's action_space
        :return: An action in the format of the action space of the fully wrapped env
        """
        if hasattr(self.env, 'wrap_action'):
            return self.reverse_action(self.env.wrap_action(inner_action))
        else:
            return self.reverse_action(inner_action)

    def reverse_action(self, action):
        raise NotImplementedError("In order to use a ReversibleActionWrapper, you need to implement a `reverse_action` function"
                                  "that is the inverse of the transformation performed on an action that comes into the wrapper")

class ActionVectorWrapper(ReversibleActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = Box(low=-1.0499999523162842, high=1.0499999523162842, shape=(64,))
    
    def action(self, action):
        return {'vector':action}
    
    def reverse_action(self, action):
        print(action)
        return action['vector']

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
    #print(next(iter(data))['acts']['vector'].shape) # (10, 64)
    
    # set up post wrappers
    post_wrappers = [
        partial(helper, wrapper_cls=ObservationPOVWrapper, wrapper_kwargs={}),
        partial(helper, wrapper_cls=ActionVectorWrapper, wrapper_kwargs={}) # TODO: only enable if using obfuscated action vector
    ]

    # set up vectorized env
    venv: VecEnv = util.make_vec_env(env_name, n_envs=n_envs, post_wrappers=post_wrappers)
    
    # set up logger
    airl_logger = logger.configure(os.path.join(log_dir, 'AIRL/'))
    
    # define reward kwargs
    reward_net_kwargs = {
        'reward_hid_sizes':[128,128],
        'potential_hid_sizes':[128,128],
        'use_next_state':True,
        'discount_factor':discount_factor
    }
    
    # init airl trainer
    policy_kwargs = dict(
        features_extractor_class=VQVAEFeatureExtractor,
        features_extractor_kwargs={'vqvae_path':visual_model_path}
    )
    airl_trainer = adversarial.AIRL(
        venv,
        expert_data=dataloader,
        expert_batch_size=expert_batch_size,
        #reward_net_cls=None,
        reward_net_kwargs=reward_net_kwargs,
        #discrim_kwargs=None,
        gen_algo=sb3.PPO('MlpPolicy', venv, verbose=int(~disable_verbose), n_steps=gen_algo_steps, policy_kwargs=policy_kwargs),
        custom_logger=airl_logger
    )
    
    # train
    airl_trainer.train(total_timesteps=total_timesteps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='MineRLTreechopVectorObf-v0')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--log_dir', type=str, default='./run_logs')
    parser.add_argument('--expert_batch_size', type=int, default=10)
    parser.add_argument('--n_envs', type=int, default=1)
    parser.add_argument('--total_timesteps', type=int, default=200)
    parser.add_argument('--gen_algo_steps', type=int, default=10)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--visual_model_path', type=str, default=None)
    parser.add_argument('--disable_verbose', action='store_false')
    
    args = parser.parse_args()
    
    main(**vars(args))