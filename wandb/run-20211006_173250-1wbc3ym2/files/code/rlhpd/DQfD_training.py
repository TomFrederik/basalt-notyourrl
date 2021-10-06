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
import wandb

from common.DQfD_utils import MemoryDataset, loss_function, preprocess_non_pov_obs, RewardWrapper, StateWrapper
from common.DQfD_models import QNetwork

class DummyRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, obs, vec):
        return torch.zeros_like(vec)[:,0]


def train(
    log_dir, 
    new_model_path,
    env_name, 
    reward_model,
    save_freq, 
    dataset: MemoryDataset, 
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
    
    # init optimizer
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    # init target q_net
    target_q_net = deepcopy(q_net).eval()
    
    # init env
    env = gym.make(env_name)
    env = StateWrapper(env)
    env = RewardWrapper(env, reward_model)
    obs = env.reset()
    done = False
    
    steps = 0
    while steps < train_steps:
        steps += 1
        
        while not done:
            # compute q values
            with torch.no_grad():
                q_input = {'pov': torch.as_tensor(obs['pov'])[None], 'vec': torch.as_tensor(obs['vec'])[None]}
                q_values = q_net.forward(**q_input)
                q_action = torch.argmax(q_values).item()
                
            # sample action from epsilon-greedy behavior policy
            if random() < epsilon:
                # choose a random action
                action = randint(0, q_net.num_actions)
            else:
                # choose action with highest Q value
                action = q_action
            
            # take action
            next_obs, reward, done, info = env.step(action)

            # compute next q values
            q_input = {'pov': torch.from_numpy(next_obs['pov'])[None], 'vec': torch.from_numpy(next_obs['vec'])[None]}
            next_q_values = q_net.forward(**q_input)
            next_q_action = torch.argmax(next_q_values).item()

            # compute td error
            td_error = torch.abs(reward + discount_factor * next_q_values[next_q_action] - q_values[action])
            
            # store transition
            transition = (
                obs,
                action,
                next_obs,
                reward,
                None, #n_step_state TODO
                0, #n_step_reward TODO
                td_error
            )
            dataset.add_agent_transition(transition)
            
            #########
            #########
            
            # sample a new batch from the dataset
            batch_idcs = dataset.combined_memory.sample(batch_size)
            state, next_state, n_step_state, action, reward, n_step_reward, idcs, weight, expert_mask = zip(*[dataset[idx] for idx in batch_idcs])
            pov, vec = zip(*state)
            next_pov, next_vec = zip(*next_state)
            pov = torch.from_numpy(np.array(pov))
            vec = torch.from_numpy(np.array(vec))
            next_pov = torch.from_numpy(np.array(next_pov))
            next_vec = torch.from_numpy(np.array(next_vec))
            weight = torch.from_numpy(np.array(weight))
            expert_mask = torch.from_numpy(np.array(expert_mask))

            # compute q values
            q_values = q_net.forward(pov, vec)
            next_q_values = q_net.forward(next_pov, next_vec)
            next_target_q_values = target_q_net.forward(next_pov, next_vec)
            n_step_q_values = q_net.forward(n_step_pov, n_step_vec)
            
            # compute actions
            q_actions = torch.argmax(q_values, 1)
            next_q_action = torch.argmax(next_q_values, 1)
            next_target_q_action = torch.argmax(next_target_q_values, 1)
            
            # compute td error
            updated_td_error = torch.abs(rew + discount_factor * next_q_values[next_q_action] - q_values[action])

            # zero gradients
            optimizer.zero_grad(set_to_none=True)

            ## compute loss
            # J_DQ
            J_DQ = (reward + discount_factor * next_q_values[np.arange(len(next_q_values)), next_target_q_action] - q_values[np.arange(len(q_values)), action]) ** 2
            
            # J_E
            pre_max_q = q_values + supervised_loss_margin
            pre_max_q[np.arange(len(action)), action] -= supervised_loss_margin
            J_E = torch.max(pre_max_q, dim=1)[0] - q_values[np.arange(len(action)), action]
            J_E = expert_mask * J_E # only compute it for expert actions
            
            # J_n
            J_n = torch.zeros_like(J_DQ) #TODO
            
            loss = (weights * (J_DQ + J_E + J_n)).mean()
                        
            # backward and step
            loss.backward()
            optimizer.step()
            
            # loss logging
            log_dict = {
                'Training/total_loss': loss,
                'Training/J_DQ': J_DQ.mean(),
                'Training/J_E': J_E.sum() / expert_mask.sum(),
                'Training/J_n': J_n.mean(),
                'Training/Step': steps
            }
            wandb.log(log_dict)
            
            # sample weight updating with td_error
            # (reward is always 0 in pretraining)
            dataset.update_td_errors(batch_idcs, td_error)
            
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
    
    wandb.init('DQfD_training')
    
    if reward_model_path is None:
        print('\nWARNING!: DummyRewardModel will be used! If you are not currently debugging, you should change that!\n')
    
    # set save dir
    # TODO: change log_dir to a single indentifier (no time, but maybe model version?)
    log_dir = os.path.join(log_dir, env_name, str(int(time())))
    os.makedirs(log_dir, exist_ok=True)
    
    # set device # TODO: use cuda?
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load q_net
    q_net = torch.load(model_path)

    # load reward model
    if reward_model_path is None:
        reward_model = DummyRewardModel()
    else:
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
    parser.add_argument('--env_name', default='MineRLBasaltFindCave-v0')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--new_model_path', type=str, required=True)
    parser.add_argument('--reward_model_path', type=str, default=None)#, required=True)
    parser.add_argument('--num_expert_episodes', type=int, default=100)
    parser.add_argument('--n_step', type=int, default=50)
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
    parser.add_argument('--agent_memory_capacity', type=int, default=20000)
    parser.add_argument('--train_steps', type=int, default=100000)
    
    args = parser.parse_args()
    
    main(**vars(args))
