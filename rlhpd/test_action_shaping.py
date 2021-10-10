import gym
import minerl

from common.action_shaping import find_cave_action_complex

env = gym.make('MineRLBasaltFindCave-v0')
action = env.action_space.sample()
action_final, index = find_cave_action_complex(action)

print(f'{action = }')
print(f'{action_final = }')
print(f'{index = }')
