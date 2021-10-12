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

house_inv_items = [
    'acacia_door',
    'acacia_fence',
    'cactus',
    'cobblestone',
    'dirt',
    'fence',
    'flower_pot',
    'glass',
    'ladder',
    'log#0',
    'log#1',
    'log2#0',
    'planks#0',
    'planks#1',
    'planks#4',
    'red_flower',
    'sand',
    'sandstone#0',
    'sandstone#2',
    'sandstone_stairs',
    'snowball',
    'spruce_door',
    'spruce_fence',
    'stone_axe',
    'stone_pickaxe',
    'stone_stairs',
    'torch',
    'wooden_door',
    'wooden_pressure_plate'
    ]
house_equip_items = house_inv_items + ['none', 'other']
house_equip_items.sort()
house_inv_items.sort()

INV_TO_IDX_HOUSE = {item: i for i,item in enumerate(house_inv_items)}
EQUIP_TO_IDX_HOUSE = {item: i for i,item in enumerate(house_equip_items)}

def preprocess_non_pov_obs(state, env_name):
    """
    Go from two dictionaries: state['equipped_items'] and state['inventory']
    to a flattened one-hot(ish) vector of the form:
    [
        damage, maxDamage,                                              # Equip stats
        isAirEquipped, isBucketEquipped, ... , isWheatSeedsEquipped,    # One-hot vector for equippable items
        numBuckets, numCarrots, ... , numWheatSeeds,                    # Count for inventory items
    ]
    """
    # add damage vector to inv_obs
    inv_obs = [state['equipped_items']['mainhand']['damage'], state['equipped_items']['mainhand']['maxDamage']]
    
    # get number of inventory items to one-hot encoded 'equip' type
    inv = list(state['inventory'].values())
    num_equippable_items = len(inv) + 3 # in addition to inv items, we have 'air', 'other' and 'none'
    equip = [0] * num_equippable_items
    if env_name == 'MineRLBasaltBuildVillageHouse-v0':
        equip[EQUIP_TO_IDX_HOUSE[state['equipped_items']['mainhand']['type']]] = 1
    else:
        equip[EQUIP_TO_IDX[state['equipped_items']['mainhand']['type']]] = 1
    
    # add equip type one-hot vector to inv_obs
    inv_obs.extend(equip)
    
    # add inventory vector to inv_obs
    inv_obs.extend(inv)
    
    return np.array(inv_obs).astype(np.float32)

def preprocess_pov_obs(state):
    return einops.rearrange(state['pov'], 'h w c -> c h w').astype(np.float32) / 255

def preprocess_state(state, env_name):
    # Append a new flattened 'vec' attribute to the state; 'vec' contains the same info
    # as state['equipped_items'] and state['inventory'], we leave those in only for
    # debugging purposes (they will not be used by the agent)
    new_state = {}
    new_state['vec'] = preprocess_non_pov_obs(state, env_name)
    new_state['pov'] = preprocess_pov_obs(state)
    return new_state

class StateWrapper(gym.ObservationWrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        self.env = env
        self.env_name = env_name
        dummy_vec = preprocess_non_pov_obs(self.observation_space.sample(), self.env_name)
        self.observation_space = gym.spaces.Dict({
            'pov':gym.spaces.Box(0, 1, (3,64,64)), 
            'vec':gym.spaces.Box(-np.inf, np.inf, dummy_vec.shape)
        })
        
    def observation(self, obs):
        obs = preprocess_state(obs, self.env_name)
        return obs

if __name__ == "__main__":
    # Run test
    test_state = {
        "equipped_items": {
            "mainhand": {
                "damage": 0,
                "maxDamage": 0,
                "type": "none"
            }
        },
        "inventory": {
            "bucket": 0,
            "carrot": 0,
            "cobblestone": 0,
            "fence": 0,
            "fence_gate": 0,
            "snowball": 0,
            "stone_pickaxe": 0,
            "stone_shovel": 0,
            "water_bucket": 0,
            "wheat": 0,
            "wheat_seeds": 3
        }
    }
    expected_vec = np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3], dtype=np.float32)
    assert np.all(preprocess_non_pov_obs(test_state, 'MineRLBasaltFindCave-v0') == expected_vec)