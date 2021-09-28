import random
from collections import deque, namedtuple
import einops
import numpy as np
from torch.utils.data import Dataset
import minerl
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'n_step_state', 'n_step_reward', 'td_error'))

# angles of rotation above which we consider camera actions (TODO: arbitrary values, probably need tweaking)
PITCH_MARGIN=5
YAW_MARGIN=5
# action priorities for each task
ACTION_PRIORITIES = {
    'MineRLBasaltFindCave-v0': {
        'attack-forward-jump': 0, 'camera_down': 1, 'camera_left': 1,
        'camera_right': 1, 'camera_up': 1, 'afj-down': 2, 'afj-left': 2,
        'afj-right': 2, 'afj-up': 2, 'equip': 3, 'use': 4
    }
}

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
        return np.concatenate([self.memory_dict['expert'].memory, self.memory_dict['agent'].memory])[idx]

    def sample(self, batch_size):
        idcs = np.random.choice(np.arange(len(self)), size=batch_size, replace=False, p=self.weights)
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
        self.combined_memory = CombinedMemory(agent_memory_capacity, n_step, discount_factor, p_offset, PER_exponent, IS_exponent)
        self.load_expert_demo(env_name, data_dir, num_expert_episodes)
        self.env_name = env_name
        
    def __len__(self):
        return len(self.combined_memory)
    
    def __getitem__(self, idx):
        state, action, next_state, reward, n_step_state, n_step_reward, td_error = self.combined_memory[idx]
        
        action = self._preprocess_action(action)
        
        pov = einops.rearrange(state['pov'], 'h w c -> c h w').astype(np.float32) / 255
        next_pov = einops.rearrange(next_state['pov'], 'h w c -> c h w').astype(np.float32) / 255
        n_step_pov = einops.rearrange(n_step_state['pov'], 'h w c -> c h w').astype(np.float32) / 255

        inv = self._preprocess_other_obs(state)
        next_inv = self._preprocess_other_obs(next_state)
        n_step_inv = self._preprocess_other_obs(n_step_state)

        reward = reward.astype(np.float32)
        n_step_reward = n_step_reward.astype(np.float32)
        
        weight = self.weights[idx]

        return (pov, inv), (next_pov, next_inv), (n_step_pov, n_step_inv), action, reward, n_step_reward, idx, weight

    def _preprocess_action(self, action):
        '''
        Returns the shaped action, depending on the environment
        '''
        return {
            'MineRLBasaltFindCave-v0':find_cave_action,
            'MineRLBasaltMakeWaterfall-v0':make_waterfall_action,
            'MineRLBasaltCreateVillageAnimalPen-v0':create_pen_action,
            'MineRLBasaltBuildVillageHouse-v0':build_house_action
        }[self.env_name](action)
        
    
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

            obs, act, rewards, _ = zip(*data.load_data(trajectory_name))

            td_errors = np.ones_like(rewards)
            
            actions = self._preprocess_action(act)

            # add episode to memory
            self.combined_memory.add_episode(obs, actions, rewards, td_errors, memory_id='expert')

        print('\nLoaded ',len(self.combined_memory.memory_dict['expert']),' expert samples!')


def preprocess_non_pov_obs(state):
    # add damage vector to inv_obs
    inv_obs = [state['equipped_items']['mainhand']['damage'], state['equipped_items']['mainhand']['maxDamage']]
    
    # get number of inventory items to one-hot encoded 'equip' type
    inv = list(state['inventory'].values())
    num_inv_items = len(inv) + 3 # in addition to inv items, we have 'air', 'other' and 'none'
    equip = [0] * num_inv_items
    equip[state['equipped_items']['mainhand']['type']] = 1
    
    # add equip type one-hot vector to inv_obs
    inv_obs.extend(equip)
    
    # add inventory vector to inv_obs
    inv_obs.extend(inv)
    
    return np.array(inv_obs).astype(np.float32)



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



'''
Define the action shaping functions, which take the original action
and map it to the shaped action
'''
def get_cam_actions_shaped(delta_pitch, delta_yaw, pitch_margin, yaw_margin,
                           move_left, move_right, consider_move=True):
    '''
    Transform a single item camera action --> 'camera': [float, float] into a 4-item dict of
    camera actions -->  {'camera_down': bool, 'camera_left': bool, 'camera_right': bool, 'camera_up': bool}
    If consider_move==True then it will also convert lateral movements to camera rotations
    '''
    cam_actions_shaped = {'camera_down': False, 'camera_left': False,
                          'camera_right': False, 'camera_up': False}
    if delta_pitch < -pitch_margin:  # camera down
        cam_actions_shaped['camera_down'] = True
    elif delta_pitch > pitch_margin:  # camera up
        cam_actions_shaped['camera_up'] = True
    if delta_yaw < -yaw_margin:  # camera left
        cam_actions_shaped['camera_left'] = True or (consider_move and bool(move_left))
    elif delta_yaw > yaw_margin:  # camera right
        cam_actions_shaped['camera_right'] = True or (consider_move and bool(move_right))
    return cam_actions_shaped

def remove_actions(actions, actions_to_remove=[]):
    '''
    Remove dict items by a given list of keys
    '''
    for act in actions_to_remove:
        del actions[act]
    return actions

def insert_actions(actions, actions_to_insert, key_ref=None):
    '''
    Insert OrderedDict items into another OrderedDict after a specified position given by a reference key
    If no reference key is given, it will insert the items at the first position
    '''
    new_actions = actions.__class__()
    if not key_ref:
        for k, v in actions_to_insert.items():
            new_actions[k] = v
    for key, value in actions.items():
        new_actions[key] = value
        if key_ref:
            if key == key_ref:
                for k,v in actions_to_insert.items():
                    new_actions[k] = v
    actions.clear()
    actions.update(new_actions)
    return actions

def combine_actions(actions, ref_key, new_key, keys_to_avoid):
    '''
    Combines a certain number of actions into a smaller "summary" of them.
    A reference key has to be present for the summary to be performed.
    The new key is the name of the new dict "summary" item. The keys to avoid
    are the keys of the input dict that are not going to be summarized
    '''
    if ref_key in actions:
        new_item = {new_key: any([v for k,v in actions.items()
                              if not k in keys_to_avoid])}
        combined_actions = insert_actions(actions, new_item)
    else:
        return actions

    return combined_actions

def combine_actions_multi(actions, ref_key, new_keys, old_keys, keys_to_avoid):
    '''
    Same as combine_actions but for multiple items
    '''
    k3 = []
    for k1, k2 in zip(new_keys, old_keys):
        keys_to_avoid += k3
        keys_to_avoid.remove(k2)
        actions = combine_actions(
            actions, ref_key=ref_key,
            new_key=k1,
            keys_to_avoid=keys_to_avoid)
        k3 += [k2] + [k1]

    return actions

def shape_equip(actions):
    '''
    Shape the equip action space
    '''
    actions['equip'] = 0 if (actions['equip'] == 'none') else 1
    # TODO: extend to rest of items and tasks

    return actions

def prioritize_actions(actions, act_prior):
    '''
    Prioritize actions depending on a ruleset. E.g.: if the agent rotates the
    camera to the left at the same time that pressing "equip", we prioritize
    the "equip" action
    '''
    for k,v in actions.items():
        actions[k] = int(actions[k]) * 10**(act_prior[k] + 1)
    max_action = max([v for v in actions.values()])
    for k,v in actions.items():
        actions[k] = int(actions[k] // (max_action if max_action else 1))

    return actions

def index_actions(actions, default):
    '''
    Return an integer given an OrderedDict that can only contain a single active action
    If no active actions, it defaults to a selected commonly useful action
    '''
    if not any([v for k, v in actions.items()]):
        actions[list(actions.keys())[default]] = 1
        return actions, default
    return actions, np.argmax([v for k, v in actions.items()])

def find_cave_action(action):
    # combine attack, forward and jump into a single action
    action_combined = combine_actions(action, ref_key='forward', new_key='attack-forward-jump',
                                      keys_to_avoid=['camera', 'equip', 'left', 'right', 'use'])
    # action['camera']=[float, float] ----> action['camera_down']= {0,1}, action['camera_left']= {0,1} , etc.
    cam_actions_shaped = get_cam_actions_shaped(*action['camera'],
                                                PITCH_MARGIN, YAW_MARGIN,
                                                action['left'], action['right'])
    # insert shaped camera actions
    action_withcam = insert_actions(action_combined, cam_actions_shaped, 'back')
    # remove actions that are not needed
    action_withcam_lean = remove_actions(
        action_withcam,
        ['attack', 'back', 'camera', 'forward', 'jump', 'left', 'right', 'sprint', 'sneak'])
    # add equip action shaping
    action_withcam_lean_equipped = shape_equip(action_withcam_lean)
    # combine attack-forward-jump action with camera rotation actions
    action_final = combine_actions_multi(
        action_withcam_lean_equipped,
        ref_key='attack-forward-jump',
        new_keys=['afj-down', 'afj-left', 'afj-right', 'afj-up'],
        old_keys=['camera_down', 'camera_left', 'camera_right', 'camera_up'],
        keys_to_avoid=['attack-forward-jump', 'camera_down', 'camera_left',
                       'camera_right', 'camera_up', 'equip', 'use'])
    # prioritize actions
    action_final_prioritized = prioritize_actions(
        action_final, act_prior=ACTION_PRIORITIES['MineRLBasaltFindCave-v0'])
    # index actions
    action_final_prioritized_indexed, index = index_actions(action_final_prioritized, 4)

    return action_final_prioritized_indexed, index

def make_waterfall_action(action):
    #TODO
    raise NotImplementedError

def build_house_action(action):
    #TODO
    raise NotImplementedError

def create_pen_action(action):
    #TODO
    raise NotImplementedError