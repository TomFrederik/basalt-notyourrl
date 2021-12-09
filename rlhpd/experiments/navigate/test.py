# from stable_baselines3 import PPO
# from stable_baselines3.common.envs import SimpleMultiObsEnv
# import gym
# from gym import spaces
# import numpy as np

# class ObservationWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         self.CHANGE_THIS_NAME = 'img'
#         self.observation_space = spaces.Dict({
#             self.CHANGE_THIS_NAME: spaces.Box(0, 255, (64, 64, 1), np.uint8), 
#             'vec': spaces.Box(0.0, 1.0, (5,), np.float64),
#         })

#     def step(self, action):
#         next_state, reward, done, info = self.env.step(action)
#         next_state = {
#             self.CHANGE_THIS_NAME: next_state['img'],
#             'vec': next_state['vec'],
#         }
#         return next_state, reward, done, info

# # Stable Baselines provides SimpleMultiObsEnv as an example environment with Dict observations
# env = SimpleMultiObsEnv(random_start=False)
# env = ObservationWrapper(env)
# model = PPO("MultiInputPolicy", env, verbose=1)
# model.learn(total_timesteps=10)



# import gym
# import highway_env
# import numpy as np
# from gym import spaces
# from stable_baselines3 import PPO

# class ObservationWrapper(gym.Wrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         self.env = env
#         self.observation_space = spaces.Dict({
#             'achieved_goal': spaces.Box(-np.inf, np.inf, (6,), np.float64), 
#             'desired_goal': spaces.Box(-np.inf, np.inf, (6,), np.float64),
#             'observation': spaces.Box(-np.inf, np.inf, (6,), np.float64)
#         })

#     def step(self, action):
#         next_state, reward, done, info = self.env.step(action)
#         next_state = {
#             'achieved_goal': next_state['achieved_goal'],
#             'desired_goal': next_state['desired_goal'],
#             'observation': next_state['observation'],
#         }
#         return next_state, reward, done, info

# env = gym.make("parking-v0")
# env = ObservationWrapper(env)
# model = PPO("MultiInputPolicy", env)
# model.learn(total_timesteps=10)



import gym
import minerl
import numpy as np
from gym import spaces
from stable_baselines3 import PPO


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = spaces.Box(low=-1.0499999523162842, high=1.0499999523162842, shape=(64,))
    
    def action(self, act):
        return {
            'vector': act,
        }

env = gym.make('MineRLNavigateDenseVectorObf-v0')
env = ActionWrapper(env)
# env = ObservationWrapper(env)
model = PPO("MultiInputPolicy", env)
model.learn(total_timesteps=10)