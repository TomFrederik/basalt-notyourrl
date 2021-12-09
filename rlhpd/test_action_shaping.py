import gym
import minerl

from common.action_shaping import action_shaping_complex, reverse_action_shaping_complex
from common.DQfD_utils import RewardActionWrapper
from DQfD_training import DummyRewardModel
from common.state_shaping import StateWrapper
env_name = 'MineRLBasaltFindCave-v0'
env = gym.make(env_name)
env = StateWrapper(env)
# forward
print('\nForward:')
action = env.action_space.sample()
action['equip'] = 'snowball'
print(f'{action = }')
action_final, index = action_shaping_complex(action, env_name)
print(f'{action_final = }')
env = RewardActionWrapper(env_name, env, DummyRewardModel())

# backward
print('\nBackward:')
action = reverse_action_shaping_complex(index, env_name)
print(f'{action = }')
