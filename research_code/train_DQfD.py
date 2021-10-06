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

from DQfD_utils import MemoryDataset, loss_function, preprocess_non_pov_obs, EnvironmentWrapper
from DQfD_models import QNetwork

def train(
    log_dir, 
    new_model_path,
    env_name, 
    reward_model,
    save_freq, 
    dataset, 
    discount_factor, 
    q_net, 
    train_steps, 
    batch_size, 
    supervised_loss_margin, 
    lr, 
    weight_decay, 
    update_freq, 
    epsilon
):
    
    # init tensorboard writer
    writer = SummaryWriter(log_dir)
    
    # init optimizer
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    # init target q_net
    target_q_net = deepcopy(q_net).eval()
    
    # init env
    env = gym.make(env_name)
    env = EnvironmentWrapper(env, reward_model) #TODO implement this
    obs = env.reset()
    done = False
    
    steps = 0
    while steps < train_steps:
        steps += 1
        
        while not done:
            # sample action from epsilon-greedy behavior policy
            if random() < epsilon:
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
        loss = loss_function(
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
        loss.backward()
        optimizer.step()
        
        # loss logging
        writer.add_scalar('Pretraining/J_E', J_E, global_step=steps)
        
        # sample weight updating with td_error
        # (reward is always 0 in pretraining)
        updated_td_errors = torch.abs(discount_factor * next_q_values - cur_q_values.detach())
        dataset.update_td_errors(batch_idcs, updated_td_errors)
        
        if steps % save_freq == 0:
            print('Saving model...')
            torch.save(q_net, new_model_path)
        
        if steps % update_freq == 0:
            print('Updating target model...')
            target_q_net = deepcopy(q_net)
            
    return q_net

def main(env_name, train_steps, save_freq, model_path, new_model_path, reward_model_path,
         lr, n_step, agent_memory_capacity, discount_factor, epsilon, batch_size, num_expert_episodes, data_dir, log_dir,
         PER_exponent, IS_exponent_0, agent_p_offset, expert_p_offset, weight_decay, supervised_loss_margin, update_freq):
    
    # set save dir
    # TODO: change log_dir to a single indentifier (no time, but maybe model version?)
    log_dir = os.path.join(log_dir, env_name, str(int(time())))
    os.makedirs(log_dir, exist_ok=True)
    
    # set device # TODO: use cuda?
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load q_net
    q_net = torch.load(model_path)

    # load reward model
    reward_model = torch.load(reward_model_path)
    
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

    # launch training
    q_net = train(
        log_dir,
        new_model_path,
        env_name,
        reward_model,
        save_freq,
        dataset,
        discount_factor, 
        q_net,
        train_steps,
        batch_size,
        supervised_loss_margin,
        lr,
        weight_decay,
        update_freq,
        epsilon
    )
    
    print('Training finished! Saving model...')
    torch.save(q_net, new_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechop-v0')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--new_model_path', type=str, required=True)
    parser.add_argument('--reward_model_path', type=str, required=True)
    parser.add_argument('--num_expert_episodes', type=int, default=100)
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
    parser.add_argument('--train_steps', type=int, default=100000)
    parser.add_argument('--update_freq', type=int, default=100)
    
    args = parser.parse_args()
    
    main(**vars(args))
