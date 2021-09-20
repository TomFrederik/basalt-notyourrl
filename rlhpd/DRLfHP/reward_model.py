import torch
import torch.nn as nn
import torch.nn.functional as F

"""
For the reward model, we use the same configuration as the Atari experiments in Christiano et al. (2017): 
84x84x4 stacked frames (same as the inputs to the policy) as inputs to 4 convolutional layers of 
size 7x7, 5x5, 3x3, and 3x3 with strides 3, 2, 1, 1, each having 16 filters, with leaky ReLU 
nonlinearities (α = 0.01). This is followed by a fully connected layer of size 64 and then a 
scalar output. The agent action at is not used as input as this did not improve performance. 
"""

class RewardModel(nn.Module):
    def __init__(self, num_actions, inv_dim, inv_network_dim=128, inv_feature_dim=128, n_hid=64, pov_feature_dim=128, q_net_dim=128):
        super().__init__()
        # pov feature extractor
        # 64x64
        channel = 16
        relu_negative_slope = 0.01
        self.conv = nn.Sequential(
            nn.Conv2d(3, channel, 7, stride=3),
            nn.LeakyReLU(negative_slope=relu_negative_slope, inplace=True),
            nn.Conv2d(channel, channel, 5, stride=2),
            nn.LeakyReLU(negative_slope=relu_negative_slope, inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1),
            nn.LeakyReLU(negative_slope=relu_negative_slope, inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        # apply conv net to pov_obs
        out = self.conv(obs)
        return out


reward_model = RewardModel()

def preference_loss(clip_1, clip_2, judgement):
    sum_rewards_1 = torch.sum(reward_model(clip_1))
    sum_rewards_2 = torch.sum(reward_model(clip_2))

    prefer_1 = torch.exp(sum_rewards_1) / (torch.exp(sum_rewards_1) + torch.exp(sum_rewards_2))
    prefer_2 = torch.exp(sum_rewards_2) / (torch.exp(sum_rewards_1) + torch.exp(sum_rewards_2))

    return - judgement[0] * torch.log(prefer_1) + judgement[1] * torch.log(prefer_2)

for clip_1, clip_2, judgement in annotation_buffer:
    loss += preference_loss(clip_1, clip_2, judgement)


## Synthetic preferences

obs, reward, done, info = env.step(action)

def simulate_judgement(clip_1, clip_2):
    sum_1 = sum([frame['reward'] for frame in clip_1])
    sum_2 = sum([frame['reward'] for frame in clip_2])

    if sum_1 == sum_2:
        judgement = (0.5, 0.5)
    elif sum_1 > sum_2:
        judgement = (1, 0)
    else:
        judgement = (0, 1)
    return judgement

"""
TODO

generate_clips.py
Run the policy and collect trajectories (incl rewards) -> Save those to file as .pkl
For N pairs:
    Select a pair of trajectories
    simulate judgement -> (clip1, clip2, judgement)
output: A folder of clips (.pkl containing list of (S,A,R,S,D) tuples)

"""