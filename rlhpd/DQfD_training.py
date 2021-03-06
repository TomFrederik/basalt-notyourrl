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
from torch.utils.data import Dataset
import wandb

from .common.action_shaping import INVENTORY
from .common.DQfD_models import QNetwork
from .common.DQfD_utils import MemoryDataset, loss_function, RewardActionWrapper, DummyRewardModel
from .common.reward_model import RewardModel
from .common.state_shaping import StateWrapper


def train(
    log_dir, 
    new_model_path,
    env_name, 
    reward_model,
    save_freq,
    max_env_steps, 
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
    env = RewardActionWrapper(StateWrapper(env, env_name), env_name, reward_model)
    obs = env.reset()
    done = False

    print('\n\n\n\nEnv reset.. Starting training\n\n\n\n')

    steps = 0
    while steps < train_steps:
        steps += 1
        
        estimated_episode_reward = 0
        while not done:
            print(steps)
            # compute q values
            with torch.no_grad():
                q_input = {'pov': torch.from_numpy(obs['pov'].copy())[None].to(q_net.device), 'vec': torch.from_numpy(obs['vec'].copy())[None].to(q_net.device)}
                q_values = q_net.forward(**q_input)[0]
                q_action = torch.argmax(q_values).item()
                
            # sample action from epsilon-greedy behavior policy
            if random() < epsilon:
                # choose a random action
                action = randint(0, q_net.num_actions)
            else:
                # choose action with highest Q value
                action = q_action
            
            # take action
            next_obs, reward, done, info = env.step(np.array(action))
            reward = reward[0]
            estimated_episode_reward += reward

            # compute next q values
            q_input = {'pov': torch.from_numpy(next_obs['pov'].copy())[None].to(q_net.device), 'vec': torch.from_numpy(next_obs['vec'].copy())[None].to(q_net.device)}
            next_q_values = q_net.forward(**q_input)[0]
            next_q_action = torch.argmax(next_q_values).item()

            # compute td error
            td_error = np.abs(reward + discount_factor * next_q_values[next_q_action].item() - q_values[action].item())
            
            # store transition
            transition = (
                obs,
                action,
                next_obs,
                reward,
                {'pov':np.array(0), 'vec':np.array(0)}, #n_step_state TODO
                np.array(0), #n_step_reward TODO
                td_error
            )
            dataset.add_agent_transition(transition)
            
            #########
            #########
            
            # sample a new batch from the dataset
            batch_idcs = dataset.combined_memory.sample(batch_size)
            state, next_state, n_step_state, action, reward, n_step_reward, idcs, weights, expert_mask = zip(*[dataset[idx] for idx in batch_idcs])
            pov, vec = zip(*state)
            next_pov, next_vec = zip(*next_state)
            #n_step_pov, n_step_vec = zip(*n_step_state)
            pov = torch.from_numpy(np.array(pov)).to(q_net.device)
            vec = torch.from_numpy(np.array(vec)).to(q_net.device)
            next_pov = torch.from_numpy(np.array(next_pov)).to(q_net.device)
            next_vec = torch.from_numpy(np.array(next_vec)).to(q_net.device)
            #n_step_pov = torch.from_numpy(np.array(n_step_pov))
            #n_step_vec = torch.from_numpy(np.array(n_step_vec))
            reward = torch.as_tensor(reward).to(q_net.device)
            weights = torch.as_tensor(weights).to(q_net.device)
            expert_mask = torch.as_tensor(expert_mask).to(q_net.device)

            # compute q values
            q_values = q_net.forward(pov, vec)
            next_q_values = q_net.forward(next_pov, next_vec)
            next_target_q_values = target_q_net.forward(next_pov, next_vec)
            #n_step_q_values = q_net.forward(n_step_pov, n_step_vec)
            
            # compute actions
            q_actions = torch.argmax(q_values, 1)
            next_q_action = torch.argmax(next_q_values, 1)
            next_target_q_action = torch.argmax(next_target_q_values, 1)
            
            # compute td error
            updated_td_error = torch.abs(reward + discount_factor * next_q_values[torch.arange(batch_size), next_q_action] - q_values[torch.arange(batch_size), action]).detach().cpu()

            # zero gradients
            optimizer.zero_grad(set_to_none=True)

            ## compute loss
            # J_DQ
            J_DQ = (reward + discount_factor * next_q_values[np.arange(batch_size), next_target_q_action] - q_values[np.arange(batch_size), action]) ** 2
            
            # J_E
            pre_max_q = q_values + supervised_loss_margin
            pre_max_q[np.arange(batch_size), action] -= supervised_loss_margin
            J_E = torch.max(pre_max_q, dim=1)[0] - q_values[np.arange(batch_size), action]
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
            dataset.update_td_errors(batch_idcs, updated_td_error)
            
            if steps % save_freq == 0:
                print('Saving model...')
                torch.save(q_net.state_dict(), new_model_path)
            
            if steps % update_freq == 0:
                print('Updating target model...')
                target_q_net = deepcopy(q_net)

            if steps == max_env_steps:
                print('Max env steps reached. Terminating episode and saving episode!')
                torch.save(q_net.state_dict(), new_model_path)

            steps += 1

        print(f'\nEpisode ended! Estimated reward: {estimated_episode_reward}\n')        
        wandb.log({'Training/Estimated Episode Reward': estimated_episode_reward})

        obs = env.reset()
        done = False

    return q_net
    
def main(env_name, train_steps, save_freq, model_path, new_model_path, reward_model_path,
         lr, horizon, agent_memory_capacity, discount_factor, epsilon, batch_size, num_expert_episodes, data_dir, log_dir,
         PER_exponent, IS_exponent_0, agent_p_offset, expert_p_offset, weight_decay, supervised_loss_margin, update_freq,
         n_hid, pov_feature_dim, vec_network_dim, vec_feature_dim, q_net_dim, max_env_steps):
    
    wandb.init(
        project=f'DQfD_training_{env_name}',
        # mode="disabled",
        )
    
    if reward_model_path is None:
        print('\nWARNING!: DummyRewardModel will be used! If you are not currently debugging, you should change that!\n')
    
    # set save dir
    # TODO: change log_dir to a single indentifier (no time, but maybe model version?)
    log_dir = os.path.join(log_dir, env_name, str(int(time())))
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
    
    # set device # TODO: use cuda?
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    


    # init dataset
    p_offset=dict(expert=expert_p_offset, agent=agent_p_offset)
    dataset = MemoryDataset(
        agent_memory_capacity,
        horizon,
        discount_factor,
        p_offset,
        PER_exponent,
        IS_exponent_0,
        env_name,
        data_dir,
        num_expert_episodes
    )

    # launch training
    # load q_net
    vec_sample = dataset[0][0][1]
    vec_dim = vec_sample.shape[0]
    print(f'vec_dim = {vec_dim}')
   
    # load reward model
    if reward_model_path is None:
        reward_model = DummyRewardModel()
    else:
        reward_model = RewardModel(vec_dim)
        reward_model.load_state_dict(torch.load(reward_model_path))

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
    q_net.load_state_dict(torch.load(model_path))
    q_net = q_net.to(device)

    q_net = train(
        log_dir,
        new_model_path,
        env_name,
        reward_model,
        save_freq,
        max_env_steps,
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
    torch.save(q_net.state_dict(), new_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLBasaltFindCave-v0')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--new_model_path', type=str, required=True)
    parser.add_argument('--reward_model_path', type=str, default=None)#, required=True)
    parser.add_argument('--num_expert_episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--update_freq', type=int, default=100)
    parser.add_argument('--max_env_steps', type=int, default=10000)
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
    parser.add_argument('--n_hid', type=int, default=64)
    parser.add_argument('--vec_feature_dim', type=int, default=128)
    parser.add_argument('--vec_network_dim', type=int, default=128)
    parser.add_argument('--pov_feature_dim', type=int, default=128)
    parser.add_argument('--q_net_dim', type=int, default=128)
    
    
    args = parser.parse_args()
    
    main(**vars(args))





