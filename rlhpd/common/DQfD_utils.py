import random
from collections import deque, namedtuple
from copy import deepcopy

import einops
import numpy as np
import torch
from torch.utils.data import Dataset
import minerl
import gym

from common.action_shaping import find_cave_action, make_waterfall_action, build_house_action, create_pen_action

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'n_step_state', 'n_step_reward', 'td_error'))

# set items for preprocessing
# TODO: incorporate stuff for village house
all_inv_items = ['bucket','carrot','cobblestone','fence','fence_gate','snowball','stone_pickaxe','stone_shovel','water_bucket','wheat','wheat_seeds']
all_equip_items = ['air','bucket','carrot','cobblestone','fence','fence_gate','none','other','snowball','stone_pickaxe','stone_shovel','water_bucket','wheat','wheat_seeds']
all_inv_items.sort()
all_equip_items.sort()
INV_TO_IDX = {item: i for i, item in enumerate(all_inv_items)}
EQUIP_TO_IDX = {item: i for i, item in enumerate(all_equip_items)}



class ReplayBuffer(object):

    def __init__(self, capacity, n_step, discount_factor, p_offset):
        self.n_step = n_step
        self.discount_factor = discount_factor
        self.p_offset = p_offset
        self.memory = deque([],maxlen=capacity)
        
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
        
        for t in range(len(obs)-self.n_step):
            state = obs[t]
            state = {'pov': preprocess_pov_obs(state), 'vec':preprocess_non_pov_obs(state)}
            action = actions[t]
            reward = rewards[t]
            td_error = td_errors[t]
            
            if t + self.n_step < len(obs):
                n_step_state = obs[t+self.n_step]
                n_step_reward = np.sum(rewards[t:t+self.n_step] * self.discount_array)
                next_state = obs[t+1]
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
    def __init__(self, agent_memory_capacity, n_step, discount_factor, p_offset, PER_exponent, IS_exponent):
        '''
        Class to combine expert and agent memory
        '''
        self.n_step = n_step
        self.discount_factor = discount_factor
        self.PER_exponent = PER_exponent
        self.IS_exponent = IS_exponent
        self.memory_dict = {
            'expert':ReplayBuffer(None, n_step, self.discount_factor, p_offset['expert']),
            'agent':ReplayBuffer(agent_memory_capacity, n_step, self.discount_factor, p_offset['agent'])
        }
        
    def __len__(self):
        return len(self.memory_dict['expert']) + len(self.memory_dict['agent'])
    
    def add_episode(self, obs, actions, rewards, td_errors, memory_id):
        self.memory_dict[memory_id].add_episode(obs, actions, rewards, td_errors)
    
        # recompute weights
        self._update_weights()

    def _update_weights(self):
        weights = np.array([(sars.td_error + self.memory_dict[key].p_offset) ** self.PER_exponent for key in ['expert', 'agent'] for sars in self.memory_dict[key].memory])
        #print(weights.shape)
        weights /= np.sum(weights) # = P(i)
        weights = 1 / (len(self) * weights) ** self.IS_exponent
        self.weights = weights / np.max(weights)
        
    def __getitem__(self, idx):
        if idx < len(self.memory_dict['expert'].memory):
            return self.memory_dict['expert'].memory[idx], 1
        else:
            return self.memory_dict['agent'].memory[idx-len(self.memory_dict['expert'].memory)], 0

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
        self.combined_memory = CombinedMemory(agent_memory_capacity, n_step, discount_factor, p_offset, PER_exponent, IS_exponent)
        self.load_expert_demo(env_name, data_dir, num_expert_episodes)
        
    def __len__(self):
        return len(self.combined_memory)
    
    def __getitem__(self, idx):
        state, action, next_state, reward, n_step_state, n_step_reward, td_error, expert = self.combined_memory[idx]

        processed_action = self._preprocess_action(action)

        pov = state['pov']
        next_pov = next_state['pov']
        n_step_pov = n_step_state['pov']

        vec = state['vec']
        next_vec = next_state['vec']
        n_step_vec = n_step_state['vec']

        reward = reward.astype(np.float32)
        n_step_reward = n_step_reward.astype(np.float32)
        
        weight = self.weights[idx]

        return (pov, vec), (next_pov, next_vec), (n_step_pov, n_step_vec), processed_action, reward, n_step_reward, idx, weight, expert

    def _preprocess_action(self, action):
        '''
        Returns the shaped action, depending on the environment
        '''
        # very hacky but there seems to be a modification of the action going on
        # TODO make this less hacky
        new_action = deepcopy(action)
 
        action_dict, idx = {
            'MineRLBasaltFindCave-v0':find_cave_action,
            'MineRLBasaltMakeWaterfall-v0':make_waterfall_action,
            'MineRLBasaltCreateVillageAnimalPen-v0':create_pen_action,
            'MineRLBasaltBuildVillageHouse-v0':build_house_action
        }[self.env_name](new_action)
        
        one_hot = np.array([*map(lambda x: x, action_dict.values())]).astype(np.float32)
        
        return one_hot
        
    
    def _preprocess_other_obs(self, state):
        '''
        Takes a state dict and stacks all observations that do not belong to 'pov' into a single vector and returns this vector
        '''
        return preprocess_non_pov_obs(state)
        
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

def preprocess_non_pov_obs(state):
    # add damage vector to inv_obs
    inv_obs = [state['equipped_items']['mainhand']['damage'], state['equipped_items']['mainhand']['maxDamage']]
    
    # get number of inventory items to one-hot encoded 'equip' type
    inv = list(state['inventory'].values())
    num_inv_items = len(inv) + 3 # in addition to inv items, we have 'air', 'other' and 'none'
    equip = [0] * num_inv_items
    equip[EQUIP_TO_IDX[state['equipped_items']['mainhand']['type']]] = 1
    
    # add equip type one-hot vector to inv_obs
    inv_obs.extend(equip)
    
    # add inventory vector to inv_obs
    inv_obs.extend(inv)
    
    return np.array(inv_obs).astype(np.float32)

def preprocess_pov_obs(state):
    return einops.rearrange(state['pov'], 'h w c -> c h w').astype(np.float32) / 255

def preprocess_state(state):
    return {'pov':preprocess_pov_obs(state), 'vec':preprocess_non_pov_obs(state)}

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

class StateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def observation(self, obs):
        return preprocess_state(obs)

class RewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_model):
        super().__init__(env)
        self.env = env
        self.reward_model = reward_model
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        reward = self.reward_model(next_state['pov'], next_state['vec'])
        
        return next_state, reward, done, info

