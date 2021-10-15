import random
from collections import deque, namedtuple
from copy import deepcopy
import itertools
from functools import partial

import einops
import gym
import minerl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from .action_shaping import action_shaping_complex, reverse_action_shaping_complex, INVENTORY
from .state_shaping import preprocess_state

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'n_step_state', 'n_step_reward', 'td_error'))


class ReplayBuffer(object):

    def __init__(self, capacity, n_step, discount_factor, p_offset, action_fn, process_state_fn):
        self.n_step = n_step
        self.discount_factor = discount_factor
        self.p_offset = p_offset
        self.memory = deque([],maxlen=capacity)
        self.action_fn = action_fn
        self.process_state_fn = process_state_fn
        
        self.discount_array = np.array([self.discount_factor ** i for i in range(self.n_step)])

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)
    
    def add_episode(self, obs, actions, rewards, td_errors):
        '''
        Adds all transitions within an episode to the memory.
        '''
        
        for t in tqdm(range(len(obs)-self.n_step)):
            state = self.process_state_fn(obs[t])
            action = self.action_fn(actions[t])[1]
            reward = rewards[t]
            td_error = td_errors[t]
            
            if t + self.n_step < len(obs):
                next_state = self.process_state_fn(obs[t+1])
                n_step_state = self.process_state_fn(obs[t+self.n_step])
                n_step_reward = np.sum(rewards[t:t+self.n_step] * self.discount_array)
            else:
                raise NotImplementedError(f't = {t}, len(obs) = {len(obs)}')
            self.push(
                state,
                action,
                next_state,
                reward,
                n_step_state,
                n_step_reward,
                td_error
            )


class CombinedMemory(object):
    def __init__(self, agent_memory_capacity, n_step, discount_factor, p_offset, PER_exponent, IS_exponent, action_fn, process_state_fn):
        '''
        Class to combine expert and agent memory
        '''
        self.n_step = n_step
        self.discount_factor = discount_factor
        self.PER_exponent = PER_exponent
        self.IS_exponent = IS_exponent
        self.memory_dict = {
            'expert':ReplayBuffer(None, n_step, self.discount_factor, p_offset['expert'], action_fn, process_state_fn),
            'agent':ReplayBuffer(agent_memory_capacity, n_step, self.discount_factor, p_offset['agent'], action_fn, process_state_fn)
        }
        
    def __len__(self):
        return len(self.memory_dict['expert']) + len(self.memory_dict['agent'])
    
    def add_episode(self, obs, actions, rewards, td_errors, memory_id):
        self.memory_dict[memory_id].add_episode(obs, actions, rewards, td_errors)
    
        # recompute weights
        self._update_weights()

    def _update_weights(self):
        expert_weights = np.array([(sars.td_error + self.memory_dict['expert'].p_offset) ** self.PER_exponent for sars in self.memory_dict['expert'].memory])
        agent_weights = np.array([(sars.td_error + self.memory_dict['agent'].p_offset) ** self.PER_exponent for sars in self.memory_dict['agent'].memory])
        weights = np.concatenate([expert_weights, agent_weights])
        weights /= np.sum(weights) # = P(i)
        weights = 1 / (len(self) * weights) ** self.IS_exponent
        self.weights = weights / np.max(weights)
        
    def __getitem__(self, idx):
        if idx < len(self.memory_dict['expert'].memory):
            return (*self.memory_dict['expert'].memory[idx], 1)
        else:
            return (*self.memory_dict['agent'].memory[idx-len(self.memory_dict['expert'].memory)], 0)

    def sample(self, batch_size):
        idcs = np.random.choice(np.arange(len(self)), size=batch_size, replace=False, p=self.weights/np.sum(self.weights))
        return idcs

    def update_IS_exponent(self, new_IS_exponent):
        self.IS_exponent = new_IS_exponent
    
    def update_td_errors(self, idcs, td_errors):
        for i, idx in enumerate(idcs):
            if idx < len(self.memory_dict['expert']):
                self.memory_dict['expert'].memory[idx]._replace(td_error=td_errors[i])
            else:
                self.memory_dict['agent'].memory[idx - len(self.memory_dict['expert'])]._replace(td_error=td_errors[i])
        
        self._update_weights()
        
    def add_agent_transition(self, transition):
        '''
        transition should be a tuple:
        ('state', 'action', 'next_state', 'reward', 'n_step_state', 'n_step_reward', 'td_error')
        '''
        assert len(transition) == 7
        self.memory_dict['agent'].push(*transition)

        self._update_weights()
        

 
class MemoryDataset(Dataset):
    
    def __init__(
        self, 
        agent_memory_capacity, 
        n_step, 
        discount_factor, 
        p_offset, 
        PER_exponent, 
        IS_exponent,
        env_name,
        data_dir,
        num_expert_episodes,
    ):
        '''
        Wrapper class around combined memory to make it compatible with Dataset and be used by DataLoader
        '''
        self.env_name = env_name
        straight_movements = ['forward', 'back']
        lateral_movements = ['left', 'right']
        jump = ['jump']
        attack = ['attack']
        camera = ['camera_left', 'camera_right', 'camera_up', 'camera_down']
        equip = [key for key in ['equip_' + item for item in INVENTORY[self.env_name]]]
        use = ['use']
        groups = [straight_movements, lateral_movements, jump, attack, camera, equip, use]
        all_options = list(itertools.product(*map(lambda x: x+['none'], groups)))
        action_fn = partial(action_shaping_complex, env_name=env_name, groups=groups, all_options=all_options)
        process_state_fn = partial(preprocess_state, env_name=env_name)
        self.combined_memory = CombinedMemory(agent_memory_capacity, n_step, discount_factor, p_offset, PER_exponent, IS_exponent, action_fn, process_state_fn)
        self.load_expert_demo(env_name, data_dir, num_expert_episodes)
        
    def __len__(self):
        return len(self.combined_memory)
    
    def __getitem__(self, idx):
        state, action, next_state, reward, n_step_state, n_step_reward, td_error, expert = self.combined_memory[idx]

        pov = state['pov']
        next_pov = next_state['pov']
        n_step_pov = n_step_state['pov']

        vec = state['vec']
        next_vec = next_state['vec']
        n_step_vec = n_step_state['vec']

        reward = reward.astype(np.float32)
        n_step_reward = n_step_reward.astype(np.float32)
        
        weight = self.weights[idx]

        return (pov, vec), (next_pov, next_vec), (n_step_pov, n_step_vec), action, reward, n_step_reward, idx, weight, expert

    def add_episode(self, obs, actions, rewards, td_errors, memory_id):
        self.combined_memory.add_episode(obs, actions, rewards, td_errors, memory_id)
    
    @property
    def weights(self):
        return self.combined_memory.weights

    def update_IS_exponent(self, new_IS_exponent):
        self.combined_memory.update_IS_exponent(new_IS_exponent)
    
    def update_td_errors(self, batch_idcs, updated_td_errors):
        self.combined_memory.update_td_errors(batch_idcs, updated_td_errors)

    def load_expert_demo(self, env_name, data_dir, num_expert_episodes):
        # load data
        print(f"Loading data of {env_name}...")
        data = minerl.data.make(env_name,  data_dir=data_dir)
        trajectory_names = data.get_trajectory_names()
        random.shuffle(trajectory_names)

        # Add trajectories to the data until we reach the required DATA_SAMPLES.
        for i, trajectory_name in enumerate(trajectory_names):
            if (i+1) > num_expert_episodes:
                break

            # load trajectory
            print(f'Loading {i+1}th episode...')
            obs, actions, rewards, *_ = zip(*data.load_data(trajectory_name))

            td_errors = np.ones_like(rewards)

            # add episode to memory
            self.combined_memory.add_episode(obs, actions, rewards, td_errors, memory_id='expert')

        print('\nLoaded ',len(self.combined_memory.memory_dict['expert']),' expert samples!')

    def add_agent_transition(self, transition):
        '''
        transition should be a tuple:
        ('state', 'action', 'next_state', 'reward', 'n_step_state', 'n_step_reward', 'td_error')
        '''
        self.combined_memory.add_agent_transition(transition)



def loss_function(
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
):
    J_DQ = (reward + discount_factor * best_one_step_target_value - cur_q_values[np.arange(len(action)), action]) ** 2
    
    pre_max_q = cur_q_values + supervised_loss_margin
    pre_max_q[np.arange(len(cur_expert_action)), cur_expert_action] -= supervised_loss_margin
    J_E = torch.max(pre_max_q, dim=1)[0] - cur_q_values[np.arange(len(cur_expert_action)), cur_expert_action]
    
    J_n = (n_step_reward + discount_factor ** n_step * best_n_step_target_value - cur_q_values[np.arange(len(action)), action]) ** 2
    
    loss = J_DQ + J_E + J_n
    return loss


class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.env = env
        self.reward_model = reward_model
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        reward = self.reward_model(torch.from_numpy(next_state['pov'])[None], torch.from_numpy(next_state['vec'])[None])[0]
        reward = reward.detach().cpu().numpy()
        
        return next_state, reward, done, info


class RewardActionWrapper(gym.Wrapper):
    def __init__(self, env, env_name, reward_model):
        super().__init__(env)
        self.env = env
        self.env_name = env_name
        print(f'{self.env_name = }')

        self.reward_model = reward_model
        self.action_space = gym.spaces.Discrete(11)
        straight_movements = ['forward', 'back']
        lateral_movements = ['left', 'right']
        jump = ['jump']
        attack = ['attack']
        camera = ['camera_left', 'camera_right', 'camera_up', 'camera_down']
        equip = [key for key in ['equip_' + item for item in INVENTORY[self.env_name]]]
        use = ['use']
        self.groups = [straight_movements, lateral_movements, jump, attack, camera, equip, use]
        self.all_options = list(itertools.product(*map(lambda x: x+['none'], self.groups)))
        self.equipped_snowball = False

    def step(self, action):
        # translate action to proper action dict
        new_action = reverse_action_shaping_complex(action, self.all_options)

        if self.env_name == "MineRLBasaltFindCave-v0":
            # Don't throw snowball! This tends to end the episode too soon
            # FindCave starts with snowball in hand, and does not have any other items in inventory
            # so we just never allow it to "use"
            new_action['use'] = np.array(0)

        if self.env_name == "MineRLBasaltMakeWaterfall-v0":
            # Don't throw snowball! This tends to end the episode too soon
            # We don't have access to the state, but we can keep track of whenever we take the
            # "equip snowball" action, and disallow "use" whenever that happens
            if new_action['equip'] == "snowball":
                self.equipped_snowball = True
            elif new_action['equip'] != "snowball" and new_action['equip'] != "none":
                self.equipped_snowball = False

            if self.equipped_snowball:
                new_action['use'] = np.array(0)

        if self.env_name == "MineRLBasaltBuildVillageHouse-v0":
            # Disable attacking!
            new_action['attack'] = np.array(0)

            # Replace the original choice of equipment with our curated set of objects
            if new_action['equip'] != "none":
                new_action['equip'] = np.random.choice([
                    "cobblestone",
                    "glass",
                    "wooden_door",
                    new_action['equip']
                ], p=[0.6, 0.1, 0.1, 0.2])
                print("Equipped", new_action['equip'])
            
            # # Turn harder!
            # new_action['camera'] *= 10
            # new_action['camera'][0] = 180
            # Turn around more
            if random.random() < 0.01:
                # Pitch rotation (Prefer looking down than looking up)
                new_action['camera'][0] = np.random.choice([-15., 30.])
                # Yaw rotation
                new_action['camera'][1] = np.random.choice([-30., 30.])
            # new_action['camera'][0] = np.clip(new_action['camera'][0], -180, 30)

            # Encourage more "using"
            if random.random() < 0.2:
                new_action['use'] = np.array(1)
        
        if self.env_name == 'MineRLBasaltCreateVillageAnimalPen-v0':
            # Disable attacking!
            new_action['attack'] = np.array(0)
            
            # Replace the original choice of equipment with our curated set of objects
            if new_action['equip'] != "none":
                new_action['equip'] = np.random.choice([
                    "fence",
                    "fence_gate",
                    new_action['equip']
                ], p=[0.8, 0.1, 0.1])
                print("Equipped", new_action['equip'])
            
            #elif random.random() < 

            # # Turn harder!
            # new_action['camera'] *= 10
            # new_action['camera'][0] = 180
            # Turn around more
            if random.random() < 0.01:
                # Pitch rotation (Prefer looking down than looking up)
                new_action['camera'][0] = np.random.choice([-15., 30.])
                # Yaw rotation
                new_action['camera'][1] = np.random.choice([-30., 30.])
            # new_action['camera'][0] = np.clip(new_action['camera'][0], -180, 30)

            # Encourage more "using"
            if random.random() < 0.3:
                new_action['use'] = np.array(1)
        
        next_state, reward, done, info = self.env.step(new_action)
        reward = self.reward_model(torch.from_numpy(next_state['pov'])[None], torch.from_numpy(next_state['vec'])[None])[0]
        reward = reward.detach().cpu().numpy()

        if done:
            # TODO: This doesn't actually work! How do I reset on each episode?
            self.equipped_snowball = False
        
        return next_state, reward, done, info


class DummyRewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, obs, vec):
        return torch.zeros_like(vec)[:,0]
