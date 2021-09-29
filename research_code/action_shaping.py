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
        'afj-right': 2, 'afj-up': 2, 'equip': 3, 'use': 4
    }
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

def reverse_find_cave_action(action):
    assert isinstance(action, int), f"{type(action) = }"
    
    action_dict = {
        "attack":0,
        "back":0,
        "camera":0,
        "equip":0,
        "forward":0,
        "jump":0,
        "left":0,
        "right":0,
        "sneak":0,
        "sprint":0,
        "use":0,
    }
    
    if action == 0:
        action_dict['attack'] = 1
        action_dict['forward'] = 1
        action_dict['jump'] = 1    
        action_dict['camera'] = (PITCH_MARGIN, 0) # up
    elif action == 1:
        action_dict['attack'] = 1
        action_dict['forward'] = 1
        action_dict['jump'] = 1    
        action_dict['camera'] = (0, YAW_MARGIN) # right
    elif action == 2:
        action_dict['attack'] = 1
        action_dict['forward'] = 1
        action_dict['jump'] = 1    
        action_dict['camera'] = (0, -YAW_MARGIN) # left
    elif action == 3:
        action_dict['attack'] = 1
        action_dict['forward'] = 1
        action_dict['jump'] = 1    
        action_dict['camera'] = (-PITCH_MARGIN, 0) # down
    elif action == 4:
        action_dict['attack'] = 1
        action_dict['forward'] = 1
        action_dict['jump'] = 1    
    elif action == 5:
        action_dict['camera'] = (-PITCH_MARGIN, 0) # down
    elif action == 6:
        action_dict['camera'] = (0, -YAW_MARGIN) # left
    elif action == 7:
        action_dict['camera'] = (0, YAW_MARGIN) # right
    elif action == 8:
        action_dict['camera'] = (PITCH_MARGIN, 0) # up
    elif action == 9:
        action_dict['equip'] = 1 # equip
    elif action == 10:
        action_dict['use'] = 1 # use
    
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
    
    

def reverse_make_waterfall_action(action):
    #TODO
    raise NotImplementedError

def reverse_build_house_action(action):
    #TODO
    raise NotImplementedError

def reverse_create_pen_action(action):
    #TODO
    raise NotImplementedError



class ActionWrapper(gym.Wrapper):
    def __init__(self, env, env_name):
        super().__init__(env)
        self.env = env
        self.env_name = env_name
        
    def step(self, action):
        # reverse map action one-hot to env action
        action = {'MineRLBasaltFindCave-v0':reverse_find_cave_action,
            'MineRLBasaltMakeWaterfall-v0':reverse_make_waterfall_action,
            'MineRLBasaltCreateVillageAnimalPen-v0':reverse_create_pen_action,
            'MineRLBasaltBuildVillageHouse-v0':reverse_build_house_action
        }[self.env_name](action)

        # take step
        next_state, reward, done, info = self.env.step(action)
        
        # return stuff
        return next_state, reward, done, info
