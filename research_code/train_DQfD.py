import argparse
import os
import random
from time import time
import einops
import gym
from random import random, randint
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from DQfD_utils import MemoryDataset, loss_function, preprocess_non_pov_obs
from DQfD_models import QNetwork

# epsilon for epsilon-greedy policy
EPSILON = 0.05 # TODO: put this in argparse

def train(log_dir, env_name, save_freq, dataset, discount_factor, q_net, train_steps, batch_size, supervised_loss_margin, lr, weight_decay, update_freq):
    
    # init tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # init optimizer
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    # init target q_net
    target_q_net = deepcopy(q_net).eval()
    
    # init env
    env = gym.make(env_name)
    obs = env.reset()
    done = False
    
    steps = 0
    while steps < train_steps:
        steps += 1
        
        while not done:
            # sample action from epsilon-greedy behavior policy
            if random() < EPSILON:
                # choose a random action
                action = randint(0, q_net.num_actions)
            else:
                # choose action with highest Q value
                with torch.no_grad():
                    q_values = q_net.forward(dict(pov=obs['pov'], inv=preprocess_non_pov_obs(obs)))
                    action = torch.argmax(q_values).item()
            raise NotImplementedError #TODO
        
        # get next batch
        batch_idcs = dataset.combined_memory.sample(batch_size)
        (pov, inv), (next_pov, next_inv), (n_step_pov, n_step_inv), cur_expert_action, reward, n_step_reward, idcs, weight = zip([dataset[idx] for idx in batch_idcs])

        # forward pass
        cur_q_values = q_net.forward(dict(pov=pov, inv=inv))
        with torch.no_grad():
            next_q_values = target_q_net.forward(dict(pov=next_pov, inv=next_inv))
            n_step_q_values = target_q_net.forward(dict(pov=n_step_pov, inv=n_step_inv))
            best_one_step_target_value = torch.max(next_q_values, dim=1)
            best_n_step_target_value = torch.max(n_step_q_values, dim=1)
        
        # sample 
        
        # zero gradients
        optimizer.zero_grad(set_to_none=True)

        # compute loss
        loss_function(
            reward, 
            action, 
            cur_expert_action, 
            discount_factor, 
            best_one_step_target_value, 
            cur_q_values, 
            supervised_loss_margin, 
            n_step_reward, 
            best_n_step_target_value, 
            n_step
        )
        
        
        # backward and step
        J_E.backward()
        optimizer.step()
        
        # loss logging
        writer.add_scalar('Pretraining/J_E', J_E, global_step=steps)
        
        # sample weight updating with td_error
        # (reward is always 0 in pretraining)
        updated_td_errors = torch.abs(discount_factor * next_q_values - cur_q_values.detach())
        dataset.update_td_errors(batch_idcs, updated_td_errors)
        
        if steps % save_freq == 0:
            print('Saving model...')
            torch.save(q_net.state_dict(), os.path.join(log_dir, 'model.pt'))
        
        if steps % update_freq == 0:
            print('Updating target model...')
            target_q_net = deepcopy(q_net)
            
    return q_net

def main(env_name, train_steps, save_freq,
         lr, n_step, agent_memory_capacity, discount_factor, epsilon, batch_size, num_expert_episodes, data_dir, log_dir,
         PER_exponent, IS_exponent_0, agent_p_offset, expert_p_offset, weight_decay, supervised_loss_margin, n_hid, 
         pov_feature_dim, inv_network_dim, inv_feature_dim, q_net_dim, update_freq):
    
    # set save dir
    # TODO: save config in file path?
    log_dir = os.path.join(log_dir, env_name, str(int(time())))
    os.makedirs(log_dir, exist_ok=True)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # init dataset
    p_offset=dict(expert=expert_p_offset, agent=agent_p_offset)
    dataset = MemoryDataset(
        agent_memory_capacity,
        n_step,
        discount_factor,
        p_offset,
        PER_exponent,
        IS_exponent_0,
        env_name,
        data_dir,
        num_expert_episodes
    )

    # init q net
    inv_sample = dataset[0][0][0]
    print(f'{inv_sample = }')
    inv_dim = inv_sample.shape[0]
    print(f'{inv_dim = }')
    
    action_sample = dataset[0][3]
    print(f'{action_sample}')
    num_actions = len(action_sample)
    print(f'{num_actions}')
    
    q_net_kwargs = {
        'num_actions':num_actions,
        'inv_dim':inv_dim,
        'n_hid':n_hid,
        'pov_feature_dim':pov_feature_dim,
        'inv_feature_dim':inv_feature_dim,
        'inv_network_dim':inv_network_dim,
        'q_net_dim':q_net_dim
    }
    q_net = QNetwork(**q_net_kwargs).load_state_dict(torch.load(q_net_path))
    
    # launch training
    q_net = train(
        log_dir,
        env_name,
        save_freq,
        dataset,
        discount_factor, 
        q_net,
        train_steps,
        batch_size,
        supervised_loss_margin,
        lr,
        weight_decay,
        update_freq
    )
    
    print('Training finished! Saving model...')
    torch.save(q_net.state_dict(), os.path.join(log_dir, 'model.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechop-v0')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--num_expert_episodes', type=int, default=10)
    parser.add_argument('--n_step', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--action_repeat', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--PER_exponent', type=float, default=0.4, help='PER exponent')
    parser.add_argument('--IS_exponent_0', type=float, default=0.6, help='Initial PER Importance Sampling exponent')
    parser.add_argument('--agent_p_offset', type=float, default=0.001)
    parser.add_argument('--expert_p_offset', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--supervised_loss_margin', type=float, default=0.8)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--agent_memory_capacity', type=int, default=20000)
    parser.add_argument('--train_steps', type=int, default=100)
    parser.add_argument('--n_hid', type=int, default=64)
    parser.add_argument('--inv_feature_dim', type=int, default=128)
    parser.add_argument('--inv_network_dim', type=int, default=128)
    parser.add_argument('--pov_feature_dim', type=int, default=128)
    parser.add_argument('--q_net_dim', type=int, default=128)
    parser.add_argument('--update_freq', type=int, default=100)
    
    args = parser.parse_args()
    
    main(**vars(args))
