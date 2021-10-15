import gym
import torch as th
#from basalt_utils import sb3_compat
#from basalt_baselines.bc import bc_baseline, WRAPPERS as bc_wrappers
from stable_baselines3.common.utils import get_device
import numpy as np
import os 
import json
from pathlib import Path

from rlhpd.common import state_shaping, utils
from rlhpd.common.action_shaping import INVENTORY
from rlhpd.common.DQfD_models import QNetwork
from rlhpd.common.DQfD_utils import RewardActionWrapper
from rlhpd.common.reward_model import RewardModel

# TODO:

# - import trained agent from 'train' directory

# In the MineRLAgent class:
# - load the model in the 'load_agent' method 
# - is some preprocessing necessary?
# - in the run_agent_on_episode method, replace the random agent with our model in the one-episode env interaction

# In the 'MineRLBehavioralCloningAgent' class:
# - in the 'load_agent' method, load the right agent's policy
# - in the 'run_agent_on_episode', wrap environment with wrappers from BC training


class EpisodeDone(Exception):
    pass


class Episode(gym.Env):
    """A class for a single episode."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s, r, d, i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s, r, d, i

    def wrap_env(self, wrappers):
        for wrapper, kwargs in wrappers:
            self.env = wrapper(env=self.env, **kwargs)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space


class MineRLAgent():
    """
    To compete in the competition, you are required to implement the two
    functions in this class:
        - load_agent: a function that loads e.g. network models
        - run_agent_on_episode: a function that plays one game of MineRL

    By default this agent behaves like a random agent: pick random action on
    each step.
    """

    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        with open("aicrowd.json",) as f:
            ai_crowd_json = json.load(f)
        if ai_crowd_json["tags"] == "BuildVillageHouse":
            self.cfg = utils.load_config("rlhpd/config_BuildVillageHouse.yaml")
        elif ai_crowd_json["tags"] == "MakeWaterfall":
            self.cfg = utils.load_config("rlhpd/config_MakeWaterfall.yaml")
        elif ai_crowd_json["tags"] == "FindCave":
            self.cfg = utils.load_config("rlhpd/config_FindCave.yaml")
        elif ai_crowd_json["tags"] == "CreateVillageAnimalPen":
            self.cfg = utils.load_config("rlhpd/config_VillageAnimalPen.yaml")
        else: 
            Exception("tag is wrong")
        self.env_name = self.cfg.env_task
        
        state = gym.make(self.env_name).observation_space.sample()
        inv = list(state['inventory'].values())
        vec_dim = 2 * len(inv) + 5
        print(f'vec_dim = {vec_dim}')

        num_actions = (len(INVENTORY[self.env_name]) + 1) * 360
        print(f'num_actions = {num_actions}')
        q_net_kwargs = {
            'num_actions':num_actions,
            'vec_dim':vec_dim,
            'n_hid':self.cfg.dqfd_args.n_hid,
            'pov_feature_dim':self.cfg.dqfd_args.pov_feature_dim,
            'vec_feature_dim':self.cfg.dqfd_args.vec_feature_dim,
            'vec_network_dim':self.cfg.dqfd_args.vec_network_dim,
            'q_net_dim':self.cfg.dqfd_args.q_net_dim
        }
        self.q_net = QNetwork(**q_net_kwargs)
        self.q_net.load_state_dict(th.load(f"train/{self.env_name}.pt", map_location=get_device('auto')))
        self.q_net = self.q_net.to(get_device('auto'))
        self.q_net.eval()

    def run_agent_on_episode(self, single_episode_env: Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        reward_model = RewardModel() # Just a dummy, reward model isn't needed for policy evaluation
        reward_model.eval()
        wrappers = [(state_shaping.StateWrapper, {'env_name':self.env_name}), (RewardActionWrapper, {"env_name": self.env_name, "reward_model": reward_model})]
        single_episode_env.wrap_env(wrappers)

        obs = single_episode_env.reset()
        done = False
        while not done:
            pov, vec = obs['pov'], obs['vec']
            pov = th.from_numpy(pov.copy())[None].to(get_device('auto'))
            vec = th.from_numpy(vec.copy())[None].to(get_device('auto'))
            q_values = self.q_net.forward(pov, vec)
            q_action = th.argmax(q_values).item()
            try:
                obs, reward, done, _ = single_episode_env.step(np.array(q_action))
            except EpisodeDone:
                done = True
                continue
        print("Done")
        # An implementation of a random agent
        # _ = single_episode_env.reset()
        # done = False
        # steps = 0
        # min_steps = 500
        # while not done:
        #     random_act = single_episode_env.action_space.sample()
        #     if steps < min_steps and random_act['equip'] == 'snowball':
        #         random_act['equip'] = 'air'
        #     single_episode_env.step(random_act)
        #     steps += 1

# class Agent(MineRLAgent):
#     def load_agent(self):
#         self.q_net = th.load(f"train/MineRLBasalt{os.getenv('MINERL_TRACK')}-v0.pt", map_location=th.device(get_device('auto')))
#         self.q_net.eval()
#     def run_agent_on_episode(self, single_episode_env : Episode):
#         wrappers = [(StateWrapper, {}), (RewardActionWrapper, {"env_name": f"MineRLBasalt{os.getenv('MINERL_TRACK')}-v0", 'reward_model': th.load(reward_model_path, map_location=th.device(get_device('auto')))})]
#         single_episode_env.wrap_env(wrappers)

#         obs = single_episode_env.reset()
#         done = False
#         while not done:
#             pov, vec = obs.values()
#             pov = th.from_numpy(pov.copy())[None].to(get_device('auto'))
#             vec = th.from_numpy(vec.copy())[None].to(get_device('auto'))

             

#             q_values = self.q_net()
#             action, _, _ = self.policy.forward(th.from_numpy(obs.copy()).unsqueeze(0).to(get_device('auto')))
#             try:
#                 if action.device.type == 'cuda':
#                     action = action.cpu()
#                 obs, reward, done, _ = single_episode_env.step(np.squeeze(action.numpy()))
#             except EpisodeDone:
#                 done = True
#                 continue


        
# class MineRLBehavioralCloningAgent(MineRLAgent):
#     def load_agent(self):
#         # TODO not sure how to get us to be able to load the policy from the right agent here
#         self.policy = th.load(f"train/MineRLBasalt{os.getenv('MINERL_TRACK')}-v0.pt", map_location=th.device(get_device('auto')))
#         self.policy.eval()

#     def run_agent_on_episode(self, single_episode_env : Episode):
#         # TODO Get wrappers actually used in BC training, and wrap environment with those
#         single_episode_env.wrap_env(bc_wrappers)
#         obs = single_episode_env.reset()
#         done = False
#         while not done:

#             action, _, _ = self.policy.forward(th.from_numpy(obs.copy()).unsqueeze(0).to(get_device('auto')))
#             try:
#                 if action.device.type == 'cuda':
#                     action = action.cpu()
#                 obs, reward, done, _ = single_episode_env.step(np.squeeze(action.numpy()))
#             except EpisodeDone:
#                 done = True
#                 continue
