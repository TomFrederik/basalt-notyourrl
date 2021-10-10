from collections import OrderedDict
from copy import deepcopy
import itertools

import numpy as np
import gym

# angles of rotation above which we consider camera actions (TODO: arbitrary values, probably need tweaking)
PITCH_MARGIN=5
YAW_MARGIN=5
# action priorities for each task
ACTION_PRIORITIES = {
    'MineRLBasaltFindCave-v0': {
        'attack-forward-jump': 0, 'camera_down': 1, 'camera_left': 1,
        'camera_right': 1, 'camera_up': 1, 'afj-down': 2, 'afj-left': 2,
        'afj-right': 2, 'afj-up': 2, 'equip_snowball': 3, 'use': 4
    },
}

ACTION_PRIORITIES_SIMPLE = {
    'MineRLBasaltFindCave-v0': {
        'forward': 0, 'left': 1, 'right': 1, 'back': 2, 'jump': 3, 'attack': 4,
        'camera_down': 5, 'camera_left': 5, 'camera_right': 5, 'camera_up': 5,
        'equip_snowball': 6, 'use': 7
    },
    'MineRLBasaltMakeWaterfall-v0': {
        'forward': 0, 'left': 1, 'right': 1, 'back': 2, 'jump': 3, 'attack': 4,
        'camera_down': 5, 'camera_left': 5, 'camera_right': 5, 'camera_up': 5,
        'equip_water_bucket': 6, 'equip_cobblestone': 6, 'equip_bucket': 6,
        'equip_stone_shovel': 6, 'equip_stone_pickaxe': 6,
        'equip_snowball': 7, 'use': 8
    },
    'MineRLBasaltBuildVillageHouse-v0': {
        'forward': 0, 'left': 1, 'right': 1, 'back': 2, 'jump': 3, 'attack': 4,
        'camera_down': 5, 'camera_left': 5, 'camera_right': 5, 'camera_up': 5,
        'equip_stone_pickaxe': 6, 'equip_stone_axe': 6, 'equip_cobblestone': 6,
        'equip_stone_stairs': 6, 'equip_fence': 6, 'equip_acacia_fence': 6,
        'equip_wooden_door': 6, 'equip_planks#0': 6, 'equip_log#0': 6,
        'equip_glass': 6, 'equip_acacia_door': 6,'equip_planks#4': 6,
        'equip_log2#0': 6, 'equip_wooden_pressure_plate': 6, 'equip_dirt': 6,
        'equip_sandstone#0': 6, 'equip_sandstone#2': 6, 'equip_ladder': 6,
        'equip_planks#1': 6, 'equip_log#1': 6, 'equip_spruce_door': 6, 'equip_torch': 6,
        'equip_snowball': 7, 'use': 8
    },
    'MineRLBasaltCreateVillageAnimalPen-v0': {
        'forward': 0, 'left': 1, 'right': 1, 'back': 2, 'jump': 3, 'attack': 4,
        'camera_down': 5, 'camera_left': 5, 'camera_right': 5, 'camera_up': 5,
        'equip_wheat': 6, 'equip_wheat_seeds': 6, 'equip_carrot': 6,
        'equip_fence_gate': 6, 'equip_fence': 6,
        'equip_snowball': 7, 'use': 8
    }
}

INVENTORY = {
    'MineRLBasaltFindCave-v0': [
            'snowball'
    ],
    'MineRLBasaltMakeWaterfall-v0': [
        'snowball', 'water_bucket', 'cobblestone', 'bucket', 'stone_shovel', 'stone_pickaxe'
    ],
    'MineRLBasaltBuildVillageHouse-v0': [
        'stone_pickaxe', 'stone_axe', 'cobblestone', 'stone_stairs',
        'fence', 'acacia_fence', 'wooden_door', 'planks#0', 'log#0', 'glass',
        'snowball', 'acacia_door', 'planks#4', 'log2#0', 'wooden_pressure_plate',
        'dirt', 'sandstone#0', 'sandstone#2', 'ladder', 'planks#1', 'log#1',
        'spruce_door', 'torch'
    ],
    'MineRLBasaltCreateVillageAnimalPen-v0': [
        'snowball', 'wheat', 'wheat_seeds', 'carrot', 'fence_gate', 'fence'
    ]
}


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

def shape_equip(actions, inventory):
    '''
    Shape the equip action space
    '''
    new_actions = {f'equip_{item}':0 for item in inventory}
    for k, item in zip(new_actions, inventory):
        new_actions[k] = 1 if (actions['equip'] == item) else 0

    return new_actions

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
    action_withcam = insert_actions(
        action_combined, cam_actions_shaped, 'back')

    # add equip action shaping
    equip_actions = shape_equip(action_withcam, INVENTORY['MineRLBasaltFindCave-v0'])
    action_withcam_equipped = insert_actions(
        action_withcam, equip_actions, 'camera'
    )

    # remove actions that are not needed
    action_lean = remove_actions(
        action_withcam_equipped,
        ['attack', 'back', 'camera', 'equip', 'forward', 'jump', 'left', 'right', 'sprint', 'sneak'])

    # combine attack-forward-jump action with camera rotation actions
    action_final = combine_actions_multi(
        action_lean,
        ref_key='attack-forward-jump',
        new_keys=['afj-down', 'afj-left', 'afj-right', 'afj-up'],
        old_keys=['camera_down', 'camera_left', 'camera_right', 'camera_up'],
        keys_to_avoid=['attack-forward-jump', 'camera_down', 'camera_left',
                       'camera_right', 'camera_up', 'equip', 'use'])

    # prioritize actions
    action_final_prioritized = prioritize_actions(
        action_final, act_prior=ACTION_PRIORITIES['MineRLBasaltFindCave-v0'])

    # index actions
    action_final_prioritized_indexed, index = index_actions(
        action_final_prioritized, 4)  # 4 is index for attack-forward-jump

    return action_final_prioritized_indexed, index

def find_cave_action_simple(action):
    # action['camera']=[float, float] ----> action['camera_down']= {0,1}, action['camera_left']= {0,1} , etc.
    cam_actions_shaped = get_cam_actions_shaped(*action['camera'],
                                                PITCH_MARGIN, YAW_MARGIN,
                                                action['left'], action['right'])

    # insert shaped camera actions
    action_withcam = insert_actions(
        action, cam_actions_shaped, 'back')

    # add equip action shaping
    equip_actions = shape_equip(action_withcam, INVENTORY['MineRLBasaltFindCave-v0'])
    action_withcam_equipped = insert_actions(
        action_withcam, equip_actions, 'forward'
    )

    # remove actions that are not needed
    action_final = remove_actions(
        action_withcam_equipped,
        ['camera', 'equip', 'sprint', 'sneak'])
    
    # prioritize actions
    action_final_prioritized = prioritize_actions(
        action_final, act_prior=ACTION_PRIORITIES_SIMPLE['MineRLBasaltFindCave-v0'])

    # index actions
    action_final_prioritized_indexed, index = index_actions(
        action_final_prioritized, 6)  # 6 is index for "move forward"

    return action_final_prioritized_indexed, index

def our_function(action_final):
    '''
    action final = OrderedDict({'forward': 0/1, ...})
    {forward, back, none} X {left, right, none} X {jump, none} X {attack, none} X {camera_left, camera_right, camera_up, camera_down, none} X {all_equip_options, none} X {use, none}
    3 * 3 * 2 * 2 * 5 * (N+1) * 2 = 360 * (N+1)
    '''
    straight_movements = ['forward', 'back']
    lateral_movements = ['left', 'right']
    jump = ['jump']
    attack = ['attack']
    camera = ['camera_left', 'camera_right', 'camera_up', 'camera_down']
    equip = [key for key in action_final.keys() if key.startswith('equip')]
    use = ['use']
    groups = [straight_movements, lateral_movements, jump, attack, camera, equip, use]

    new_action_final = deepcopy(action_final)

    key_tuple = []
    for group in groups:
        found = False
        for key in group:
            if action_final[key]:
                new_action_final[key] = 1
                key_tuple.append(key)
                found = True
                break
            else:
                new_action_final[key] = 0
        if not found:
            key_tuple.append('none')
    key_tuple = tuple(key_tuple)

    all_options = list(itertools.product(*map(lambda x: x+['none'], groups)))
    index = all_options.index(key_tuple)

    return new_action_final, index
    
    

def action_shaping_complex(action, env_name):
    # action['camera']=[float, float] ----> action['camera_down']= {0,1}, action['camera_left']= {0,1} , etc.
    cam_actions_shaped = get_cam_actions_shaped(*action['camera'],
                                                PITCH_MARGIN, YAW_MARGIN,
                                                action['left'], action['right'])

    # insert shaped camera actions
    action_withcam = insert_actions(
        action, cam_actions_shaped, 'back')

    # add equip action shaping
    equip_actions = shape_equip(action_withcam, INVENTORY[env_name])
    action_withcam_equipped = insert_actions(
        action_withcam, equip_actions, 'forward'
    )

    # remove actions that are not needed
    action_final = remove_actions(
        action_withcam_equipped,
        ['camera', 'equip', 'sprint', 'sneak'])
    
    #
    action_final, index = our_function(action_final)

    return action_final, index

def reverse_action_shaping_complex(index, env_name):
    straight_movements = ['forward', 'back']
    lateral_movements = ['left', 'right']
    jump = ['jump']
    attack = ['attack']
    camera = ['camera_left', 'camera_right', 'camera_up', 'camera_down']
    equip = [key for key in ['equip_' + item for item in INVENTORY[env_name]]]
    use = ['use']
    groups = [straight_movements, lateral_movements, jump, attack, camera, equip, use]

    all_options = list(itertools.product(*map(lambda x: x+['none'], groups)))
    
    active_action_keys = all_options[index]
    
    action_dict = OrderedDict([
        ("attack",np.array(0)),
        ("back",np.array(0)),
        ("camera",np.array([0,0])),
        ("equip",'none'),
        ("forward",np.array(0)),
        ("jump",np.array(0)),
        ("left",np.array(0)),
        ("right",np.array(0)),
        ("sneak",np.array(0)),
        ("sprint",np.array(0)),
        ("use",np.array(0))
    ])

    for key in active_action_keys:
        if key.startswith('equip_'):
            action_dict['equip'] = key[6:]
        elif key.startswith('camera'):
            action_dict['camera'] = CAMERA_STR_TO_ARRAY[key]
        else:
            action_dict[key] = np.array(1)
    return action_dict

CAMERA_STR_TO_ARRAY = {
    'camera_left': np.array([0, -YAW_MARGIN]).astype(np.float32),
    'camera_right': np.array([0, YAW_MARGIN]).astype(np.float32),
    'camera_up': np.array([PITCH_MARGIN, 0]).astype(np.float32),
    'camera_down': np.array([-PITCH_MARGIN, 0]).astype(np.float32)
}


def make_waterfall_action_simple(action):
    # action['camera']=[float, float] ----> action['camera_down']= {0,1}, action['camera_left']= {0,1} , etc.
    cam_actions_shaped = get_cam_actions_shaped(*action['camera'],
                                                PITCH_MARGIN, YAW_MARGIN,
                                                action['left'], action['right'])

    # insert shaped camera actions
    action_withcam = insert_actions(
        action, cam_actions_shaped, 'back')

    # add equip action shaping
    equip_actions = shape_equip(action_withcam, INVENTORY['MineRLBasaltMakeWaterfall-v0'])
    action_withcam_equipped = insert_actions(
        action_withcam, equip_actions, 'forward'
    )

    # remove actions that are not needed
    action_final = remove_actions(
        action_withcam_equipped,
        ['camera', 'equip', 'sprint', 'sneak'])
    
    # prioritize actions
    action_final_prioritized = prioritize_actions(
        action_final, act_prior=ACTION_PRIORITIES_SIMPLE['MineRLBasaltMakeWaterfall-v0'])

    # index actions
    action_final_prioritized_indexed, index = index_actions(
        action_final_prioritized, 6)  # 6 is index for "move forward"

    return action_final_prioritized_indexed, index

def build_house_action_simple(action):
    # action['camera']=[float, float] ----> action['camera_down']= {0,1}, action['camera_left']= {0,1} , etc.
    cam_actions_shaped = get_cam_actions_shaped(*action['camera'],
                                                PITCH_MARGIN, YAW_MARGIN,
                                                action['left'], action['right'])

    # insert shaped camera actions
    action_withcam = insert_actions(
        action, cam_actions_shaped, 'back')

    # add equip action shaping
    equip_actions = shape_equip(action_withcam, INVENTORY['MineRLBasaltBuildVillageHouse-v0'])
    action_withcam_equipped = insert_actions(
        action_withcam, equip_actions, 'forward'
    )

    # remove actions that are not needed
    action_final = remove_actions(
        action_withcam_equipped,
        ['camera', 'equip', 'sprint', 'sneak'])
    
    # prioritize actions
    action_final_prioritized = prioritize_actions(
        action_final, act_prior=ACTION_PRIORITIES_SIMPLE['MineRLBasaltBuildVillageHouse-v0'])

    # index actions
    action_final_prioritized_indexed, index = index_actions(
        action_final_prioritized, 6)  # 6 is index for 'move forward'

    return action_final_prioritized_indexed, index

def create_pen_action_simple(action):
    # action['camera']=[float, float] ----> action['camera_down']= {0,1}, action['camera_left']= {0,1} , etc.
    cam_actions_shaped = get_cam_actions_shaped(*action['camera'],
                                                PITCH_MARGIN, YAW_MARGIN,
                                                action['left'], action['right'])

    # insert shaped camera actions
    action_withcam = insert_actions(
        action, cam_actions_shaped, 'back')

    # add equip action shaping
    equip_actions = shape_equip(action_withcam, INVENTORY['MineRLBasaltCreateVillageAnimalPen-v0'])
    action_withcam_equipped = insert_actions(
        action_withcam, equip_actions, 'forward'
    )

    # remove actions that are not needed
    action_final = remove_actions(
        action_withcam_equipped,
        ['camera', 'equip', 'sprint', 'sneak'])
    
    # prioritize actions
    action_final_prioritized = prioritize_actions(
        action_final, act_prior=ACTION_PRIORITIES_SIMPLE['MineRLBasaltCreateVillageAnimalPen-v0'])

    # index actions
    action_final_prioritized_indexed, index = index_actions(
        action_final_prioritized, 6)  # 6 is index for 'move forward'

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

def reverse_find_cave_action(action):
    # assert isinstance(action, int), f"{type(action) = }"
    
    action_dict = OrderedDict([
        ("attack",np.array(0)),
        ("back",np.array(0)),
        ("camera",np.array([0,0])),
        ("equip",'none'),
        ("forward",np.array(0)),
        ("jump",np.array(0)),
        ("left",np.array(0)),
        ("right",np.array(0)),
        ("sneak",np.array(0)),
        ("sprint",np.array(0)),
        ("use",np.array(0))
    ])
    
    if action == 0:
        action_dict['attack'] = np.array(1)
        action_dict['forward'] = np.array(1)
        action_dict['jump'] = np.array(1)    
        action_dict['camera'] = np.array([PITCH_MARGIN, 0]).astype(np.float32) # up
    elif action == 1:
        action_dict['attack'] = np.array(1)
        action_dict['forward'] = np.array(1)
        action_dict['jump'] = np.array(1)    
        action_dict['camera'] =  np.array([0, YAW_MARGIN]).astype(np.float32) # right
    elif action == 2:
        action_dict['attack'] = np.array(1)
        action_dict['forward'] = np.array(1)
        action_dict['jump'] = np.array(1)    
        action_dict['camera'] =  np.array([0, -YAW_MARGIN]).astype(np.float32) # left
    elif action == 3:
        action_dict['attack'] = np.array(1)
        action_dict['forward'] = np.array(1)
        action_dict['jump'] = np.array(1)    
        action_dict['camera'] = np.array([-PITCH_MARGIN, 0]).astype(np.float32) # down
    elif action == 4:
        action_dict['attack'] = np.array(1)
        action_dict['forward'] = np.array(1)
        action_dict['jump'] = np.array(1)    
    elif action == 5:
        action_dict['camera'] = np.array([-PITCH_MARGIN, 0]).astype(np.float32) # down
    elif action == 6:
        action_dict['camera'] = np.array([0, -YAW_MARGIN]).astype(np.float32) # left
    elif action == 7:
        action_dict['camera'] = np.array([0, YAW_MARGIN]).astype(np.float32) # right
    elif action == 8:
        action_dict['camera'] = np.array([PITCH_MARGIN, 0]).astype(np.float32) # up
    elif action == 9:
        action_dict['equip'] = 'snowball' # equip
    elif action == 10:
        action_dict['use'] = np.array(1) # use
    
    """
    action dict: 
    OrderedDict([
        ('afj-up', 0 or 1), 
        ('afj-right', 0 or 1), 
        ('afj-left', 0 or 1), 
        ('afj-down', 0 or 1), 
        ('attack-forward-jump', 0 or 1), 
        ('camera_down', 0 or 1), 
        ('camera_left', 0 or 1), 
        ('camera_right', 0 or 1), 
        ('camera_up', 0 or 1), 
        ('equip', 0 or 1), 
        ('use', 0 or 1)
    ])
    """
    return action_dict
    
def reverse_find_cave_action_simple(action):
    # assert isinstance(action, int), f"{type(action) = }"
    
    action_dict = OrderedDict([
        ("attack",np.array(0)),
        ("back",np.array(0)),
        ("camera",np.array([0,0])),
        ("equip",'none'),
        ("forward",np.array(0)),
        ("jump",np.array(0)),
        ("left",np.array(0)),
        ("right",np.array(0)),
        ("sneak",np.array(0)),
        ("sprint",np.array(0)),
        ("use",np.array(0))
    ])
    
    if action == 0:
        action_dict['attack'] = np.array(1)
    elif action == 1:
        action_dict['back'] = np.array(1)
    elif action == 2:
        action_dict['camera'] = np.array([-PITCH_MARGIN, 0]).astype(np.float32) # down
    elif action == 3:
        action_dict['camera'] =  np.array([0, -YAW_MARGIN]).astype(np.float32) # left
    elif action == 4:
        action_dict['camera'] =  np.array([0, YAW_MARGIN]).astype(np.float32) # right
    elif action == 5:
        action_dict['camera'] = np.array([PITCH_MARGIN, 0]).astype(np.float32) # up
    elif action == 6:
        action_dict['forward'] = np.array(1)
    elif action == 7:
        action_dict['equip'] = 'snowball' # equip
    elif action == 8:
        action_dict['jump'] = np.array(1)
    elif action == 9:
        action_dict['left'] = np.array(1)
    elif action == 10:
        action_dict['right'] = np.array(1)
    elif action == 11:
        action_dict['use'] = np.array(1) # use
    
    """
    action dict: 
    OrderedDict([
        ('attack', 0),
        ('back', 0),
        ('camera_down', 0),
        ('camera_left', 0),
        ('camera_right', 0),
        ('camera_up', 0),
        ('forward', 0),
        ('equip_snowball', 0),
        ('jump', 0),
        ('left', 0),
        ('right', 0),
        ('use', 0)])
    ])
    """
    return action_dict

def reverse_make_waterfall_action_simple(action):
    # assert isinstance(action, int), f"{type(action) = }"

    action_dict = OrderedDict([
        ("attack",np.array(0)),
        ("back",np.array(0)),
        ("camera",np.array([0,0])),
        ("equip",'none'),
        ("forward",np.array(0)),
        ("jump",np.array(0)),
        ("left",np.array(0)),
        ("right",np.array(0)),
        ("sneak",np.array(0)),
        ("sprint",np.array(0)),
        ("use",np.array(0))
    ])

    if action == 0:
        action_dict['attack'] = np.array(1)
    elif action == 1:
        action_dict['back'] = np.array(1)
    elif action == 2:
        action_dict['camera'] = np.array([-PITCH_MARGIN, 0]).astype(np.float32) # down
    elif action == 3:
        action_dict['camera'] =  np.array([0, -YAW_MARGIN]).astype(np.float32) # left
    elif action == 4:
        action_dict['camera'] =  np.array([0, YAW_MARGIN]).astype(np.float32) # right
    elif action == 5:
        action_dict['camera'] = np.array([PITCH_MARGIN, 0]).astype(np.float32) # up
    elif action == 6:
        action_dict['forward'] = np.array(1)
    elif action == 7:
        action_dict['equip'] = 'snowball' # equip
    elif action == 8:
        action_dict['equip'] = 'water_bucket' # equip
    elif action == 9:
        action_dict['equip'] = 'cobblestone' # equip
    elif action == 10:
        action_dict['equip'] = 'bucket' # equip
    elif action == 11:
        action_dict['equip'] = 'stone_shovel' # equip
    elif action == 12:
        action_dict['equip'] = 'stone_pickaxe' # equip
    elif action == 13:
        action_dict['jump'] = np.array(1)
    elif action == 14:
        action_dict['left'] = np.array(1)
    elif action == 15:
        action_dict['right'] = np.array(1)
    elif action == 16:
        action_dict['use'] = np.array(1) # use
    
    """
    OrderedDict([
        ('attack', 0),
        ('back', 0),
        ('camera_down', 0),
        ('camera_left', 0),
        ('camera_right', 0),
        ('camera_up', 0),
        ('forward', 0),
        ('equip_snowball', 0),
        ('equip_water_bucket', 0),
        ('equip_cobblestone', 0),
        ('equip_bucket', 0),
        ('equip_stone_shovel', 0),
        ('equip_stone_pickaxe', 0),
        ('jump', 0),
        ('left', 0),
        ('right', 0),
        ('use', 0)
    ])
    """

def reverse_build_house_action_simple(action):
    # assert isinstance(action, int), f"{type(action) = }"

    action_dict = OrderedDict([
        ("attack",np.array(0)),
        ("back",np.array(0)),
        ("camera",np.array([0,0])),
        ("equip",'none'),
        ("forward",np.array(0)),
        ("jump",np.array(0)),
        ("left",np.array(0)),
        ("right",np.array(0)),
        ("sneak",np.array(0)),
        ("sprint",np.array(0)),
        ("use",np.array(0))
    ])

    if action == 0:
        action_dict['attack'] = np.array(1)
    elif action == 1:
        action_dict['back'] = np.array(1)
    elif action == 2:
        action_dict['camera'] = np.array([-PITCH_MARGIN, 0]).astype(np.float32) # down
    elif action == 3:
        action_dict['camera'] =  np.array([0, -YAW_MARGIN]).astype(np.float32) # left
    elif action == 4:
        action_dict['camera'] =  np.array([0, YAW_MARGIN]).astype(np.float32) # right
    elif action == 5:
        action_dict['camera'] = np.array([PITCH_MARGIN, 0]).astype(np.float32) # up
    elif action == 6:
        action_dict['forward'] = np.array(1)
    elif action == 7:
        action_dict['equip'] = 'stone_pickaxe' # equip
    elif action == 8:
        action_dict['equip'] = 'stone_axe' # equip
    elif action == 9:
        action_dict['equip'] = 'cobblestone' # equip
    elif action == 10:
        action_dict['equip'] = 'stone_stairs' # equip
    elif action == 11:
        action_dict['equip'] = 'fence' # equip
    elif action == 12:
        action_dict['equip'] = 'acacia_fence' # equip
    elif action == 13:
        action_dict['equip'] = 'wooden_door' # equip
    elif action == 14:
        action_dict['equip'] = 'planks#0' # equip
    elif action == 15:
        action_dict['equip'] = 'log#0' # equip
    elif action == 16:
        action_dict['equip'] = 'glass' # equip
    elif action == 17:
        action_dict['equip'] = 'snowball' # equip
    elif action == 18:
        action_dict['equip'] = 'acacia_door' # equip
    elif action == 19:
        action_dict['equip'] = 'planks#4' # equip
    elif action == 20:
        action_dict['equip'] = 'log2#0' # equip
    elif action == 21:
        action_dict['equip'] = 'wooden_pressure_plate' # equip
    elif action == 22:
        action_dict['equip'] = 'sandstone#0' # equip
    elif action == 23:
        action_dict['equip'] = 'sandstone#2' # equip
    elif action == 24:
        action_dict['equip'] = 'ladder' # equip
    elif action == 25:
        action_dict['equip'] = 'planks#1' # equip
    elif action == 26:
        action_dict['equip'] = 'log#1' # equip
    elif action == 27:
        action_dict['equip'] = 'spruce_door' # equip
    elif action == 28:
        action_dict['equip'] = 'torch' # equip
    elif action == 29:
        action_dict['jump'] = np.array(1)
    elif action == 30:
        action_dict['left'] = np.array(1)
    elif action == 31:
        action_dict['right'] = np.array(1)
    elif action == 32:
        action_dict['use'] = np.array(1) # use
    
    """
    OrderedDict([
        ('attack', 0),
        ('back', 0),
        ('camera_down', 0),
        ('camera_left', 0),
        ('camera_right', 0),
        ('camera_up', 0),
        ('forward', 0),
        ('equip_stone_pickaxe', 0),
        ('equip_stone_axe', 0),
        ('equip_cobblestone', 0),
        ('equip_stone_stairs', 0),
        ('equip_fence', 0),
        ('equip_acacia_fence', 0),
        ('equip_wooden_door', 0),
        ('equip_planks#0', 0),
        ('equip_log#0', 0),
        ('equip_glass', 0),
        ('equip_snowball', 0),
        ('equip_acacia_door', 0),
        ('equip_planks#4', 0),
        ('equip_log2#0', 0),
        ('equip_wooden_pressure_plate', 0),
        ('equip_dirt', 0),
        ('equip_sandstone#0', 0),
        ('equip_sandstone#2', 0),
        ('equip_ladder', 0),
        ('equip_planks#1', 0),
        ('equip_log#1', 0),
        ('equip_spruce_door', 0),
        ('equip_torch', 0),
        ('jump', 0),
        ('left', 0),
        ('right', 0),
        ('use', 0)
    ])
    """

def reverse_create_pen_action_simple(action):
    # assert isinstance(action, int), f"{type(action) = }"

    action_dict = OrderedDict([
        ("attack",np.array(0)),
        ("back",np.array(0)),
        ("camera",np.array([0,0])),
        ("equip",'none'),
        ("forward",np.array(0)),
        ("jump",np.array(0)),
        ("left",np.array(0)),
        ("right",np.array(0)),
        ("sneak",np.array(0)),
        ("sprint",np.array(0)),
        ("use",np.array(0))
    ])

    if action == 0:
        action_dict['attack'] = np.array(1)
    elif action == 1:
        action_dict['back'] = np.array(1)
    elif action == 2:
        action_dict['camera'] = np.array([-PITCH_MARGIN, 0]).astype(np.float32) # down
    elif action == 3:
        action_dict['camera'] =  np.array([0, -YAW_MARGIN]).astype(np.float32) # left
    elif action == 4:
        action_dict['camera'] =  np.array([0, YAW_MARGIN]).astype(np.float32) # right
    elif action == 5:
        action_dict['camera'] = np.array([PITCH_MARGIN, 0]).astype(np.float32) # up
    elif action == 6:
        action_dict['forward'] = np.array(1)
    elif action == 7:
        action_dict['equip'] = 'snowball' # equip
    elif action == 8:
        action_dict['equip'] = 'wheat' # equip
    elif action == 9:
        action_dict['equip'] = 'wheat_seeds' # equip
    elif action == 10:
        action_dict['equip'] = 'carrot' # equip
    elif action == 11:
        action_dict['equip'] = 'fence_gate' # equip
    elif action == 12:
        action_dict['equip'] = 'fence' # equip
    elif action == 13:
        action_dict['jump'] = np.array(1)
    elif action == 14:
        action_dict['left'] = np.array(1)
    elif action == 15:
        action_dict['right'] = np.array(1)
    elif action == 16:
        action_dict['use'] = np.array(1) # use
    
    """
    OrderedDict([
        ('attack', 0),
        ('back', 0),
        ('camera_down', 0),
        ('camera_left', 0),
        ('camera_right', 0),
        ('camera_up', 0),
        ('forward', 0),
        ('equip_snowball', 0),
        ('equip_wheat', 0),
        ('equip_wheat_seeds', 0),
        ('equip_carrot', 0),
        ('equip_fence_gate', 0),
        ('equip_fence', 0),
        ('jump', 0),
        ('left', 0),
        ('right', 0),
        ('use', 0)
    ])
    """

def reverse_build_house_action(action):
    #TODO
    raise NotImplementedError

def reverse_create_pen_action(action):
    #TODO
    raise NotImplementedError

from gym import spaces

class ActionWrapper(gym.Wrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        raise NotImplementedError("DEPRECATED! Use RewardActionWrapper instead!")
        self.env = env
        self.env_name = env_name
        self.action_space = spaces.Discrete(11)
        
    def step(self, action):
        # reverse map action one-hot to env action
        new_action = {'MineRLBasaltFindCave-v0':reverse_find_cave_action_simple,
            'MineRLBasaltMakeWaterfall-v0':reverse_make_waterfall_action_simple,
            'MineRLBasaltCreateVillageAnimalPen-v0':reverse_create_pen_action,
            'MineRLBasaltBuildVillageHouse-v0':reverse_build_house_action
        }[self.env_name](action)

        # take step
        next_state, reward, done, info = self.env.step(new_action)
        
        # return stuff
        return next_state, reward, done, info
