import argparse
import os
import random
from time import time
import einops
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import gym
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import wandb

from .common.DQfD_utils import MemoryDataset
from .common.DQfD_models import QNetwork
from .common.action_shaping import INVENTORY

def pretrain(
    log_dir, 
    model_path, 
    save_freq, 
    dataset, 
    discount_factor, 
    q_net, 
    pretrain_steps, 
    batch_size, 
    supervised_loss_margin, 
    lr, 
    weight_decay, 
    update_freq
):
    
    # init optimizer
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    # init target q_net
    target_q_net = deepcopy(q_net).eval()
    
    steps = 0
    while steps < pretrain_steps:
        steps += 1
        
        # get next batch
        batch_idcs = dataset.combined_memory.sample(batch_size)
        state, next_state, _, cur_expert_action, _, _, idcs, weight, _ = zip(*[dataset[idx] for idx in batch_idcs])
        pov, vec = zip(*state)
        next_pov, next_vec = zip(*next_state)
        pov = torch.from_numpy(np.array(pov))
        vec = torch.from_numpy(np.array(vec))
        next_pov = torch.from_numpy(np.array(next_pov))
        next_vec = torch.from_numpy(np.array(next_vec))
        weight = torch.from_numpy(np.array(weight))
        cur_expert_action = np.array(cur_expert_action)

        # forward pass
        cur_q_values = q_net.forward(pov, vec)
        with torch.no_grad():
            next_q_values = target_q_net.forward(next_pov, next_vec)
        
        # zero gradients
        optimizer.zero_grad(set_to_none=True)

        # compute loss
        pre_max_q = cur_q_values + supervised_loss_margin
        pre_max_q[np.arange(len(cur_expert_action)), cur_expert_action] -= supervised_loss_margin
        J_E = torch.max(pre_max_q, dim=1)[0] - cur_q_values[np.arange(len(cur_expert_action)), cur_expert_action]
        J_E = (weight * J_E).mean()

        expert_q_values = cur_q_values[np.arange(len(cur_expert_action)), cur_expert_action].mean()
        other_q_values = cur_q_values.mean() - expert_q_values * q_net.num_actions
        
        # backward and step
        J_E.backward()
        optimizer.step()
        
        # loss logging
        log_dict = {
            'Pretraining/J_E':J_E,
            'Pretraining/ExpertQValues': expert_q_values,
            'Pretraining/OtherQValues': other_q_values,
            'Pretraining/expert_actions': wandb.Histogram(cur_expert_action),
            'Pretraining/Step': steps
        }
        wandb.log(log_dict)
        
        # sample weight updating with td_error
        # (reward is always 0 in pretraining)
        updated_td_errors = torch.abs(discount_factor * next_q_values - cur_q_values.detach())
        dataset.update_td_errors(batch_idcs, updated_td_errors)
        
        if steps % save_freq == 0:
            print(f'Saving model to {model_path} ...')
            torch.save(q_net.state_dict(), model_path)
        
        if steps % update_freq == 0:
            print('Updating target model...')
            target_q_net = deepcopy(q_net)
            
    return q_net

def main(env_name, pretrain_steps, save_freq, model_path, load_from_checkpoint,
         lr, horizon, discount_factor, epsilon, batch_size, num_expert_episodes, data_dir, log_dir,
         PER_exponent, IS_exponent_0, agent_p_offset, expert_p_offset, weight_decay, supervised_loss_margin, n_hid, 
         pov_feature_dim, vec_network_dim, vec_feature_dim, q_net_dim, update_freq):
    
    config = dict(
        load_from_checkpoint=load_from_checkpoint,
        n_hid=n_hid,
        q_net_dim=q_net_dim,
        pov_feature_dim=pov_feature_dim,
        vec_feature_dim=vec_feature_dim,
        supervised_loss_margin=supervised_loss_margin
    )
    wandb.init(
        project=f"DQfD_pretraining_{env_name}",
        # mode="disabled",
        tags=['basalt'],
        config=config
    )
    
    # set save dir
    # TODO: change log_dir to a single indentifier (no time, but maybe model version?)
    log_dir = os.path.join(log_dir, env_name, str(int(time())))
    if model_path is None:
        model_path = os.path.join(log_dir, 'Q_0.pth')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True) # TODO make this prettier
    
    if load_from_checkpoint:
        print(f'\nLoading model from {model_path}')
        new_model_path = model_path[:-4] + '_new.pth'
    else:
        new_model_path = model_path

    print(f'\nModel will be saved to {new_model_path}\n')


    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # init dataset
    p_offset=dict(expert=expert_p_offset, agent=agent_p_offset)
    dataset = MemoryDataset(
        0, #agent_memory_capacity is not needed
        horizon,
        discount_factor,
        p_offset,
        PER_exponent,
        IS_exponent_0,
        env_name,
        data_dir,
        num_expert_episodes
    )

    # init q net
    vec_sample = dataset[0][0][1]
    vec_dim = vec_sample.shape[0]
    print(f'vec_dim = {vec_dim}')

    num_actions = (len(INVENTORY[env_name]) + 1) * 360
    print(f'num_actions = {num_actions}')
    
    q_net_kwargs = {
        'num_actions':num_actions,
        'vec_dim':vec_dim,
        'n_hid':n_hid,
        'pov_feature_dim':pov_feature_dim,
        'vec_feature_dim':vec_feature_dim,
        'vec_network_dim':vec_network_dim,
        'q_net_dim':q_net_dim
    }
    q_net = QNetwork(**q_net_kwargs)
    if load_from_checkpoint:
        q_net.load_state_dict(torch.load(model_path))
    
    # launch pretraining
    q_net = pretrain(
        log_dir,
        new_model_path,
        save_freq,
        dataset,
        discount_factor, 
        q_net,
        pretrain_steps,
        batch_size,
        supervised_loss_margin,
        lr,
        weight_decay,
        update_freq
    )
    
    print(f'Training finished! Saving model to {model_path} ...')
    torch.save(q_net.state_dict(), model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLBasaltFindCave-v0')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/basalt-notyourrl/run_logs')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--load_from_checkpoint', action='store_true')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--num_expert_episodes', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--update_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--PER_exponent', type=float, default=0.4, help='PER exponent')
    parser.add_argument('--IS_exponent_0', type=float, default=0.6, help='Initial PER Importance Sampling exponent')
    parser.add_argument('--agent_p_offset', type=float, default=0.001)
    parser.add_argument('--expert_p_offset', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--supervised_loss_margin', type=float, default=0.8)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--pretrain_steps', type=int, default=10000)
    parser.add_argument('--n_hid', type=int, default=64)
    parser.add_argument('--vec_feature_dim', type=int, default=128)
    parser.add_argument('--vec_network_dim', type=int, default=128)
    parser.add_argument('--pov_feature_dim', type=int, default=128)
    parser.add_argument('--q_net_dim', type=int, default=128)
    
    args = parser.parse_args()
    
    main(**vars(args))
