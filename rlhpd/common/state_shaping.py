import numpy as np
import einops
import gym

# set items for preprocessing
# TODO: incorporate stuff for village house
all_inv_items = ['bucket','carrot','cobblestone','fence','fence_gate','snowball','stone_pickaxe','stone_shovel','water_bucket','wheat','wheat_seeds']
all_equip_items = ['air','bucket','carrot','cobblestone','fence','fence_gate','none','other','snowball','stone_pickaxe','stone_shovel','water_bucket','wheat','wheat_seeds']
all_inv_items.sort()
all_equip_items.sort()
INV_TO_IDX = {item: i for i, item in enumerate(all_inv_items)}
EQUIP_TO_IDX = {item: i for i, item in enumerate(all_equip_items)}

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

class StateWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def observation(self, obs):
        return preprocess_state(obs)