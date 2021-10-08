import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """
    For the reward model, we use the same configuration as the Atari experiments in Christiano et al. (2017): 
    84x84x4 stacked frames (same as the inputs to the policy) as inputs to 4 convolutional layers of 
    size 7x7, 5x5, 3x3, and 3x3 with strides 3, 2, 1, 1, each having 16 filters, with leaky ReLU 
    nonlinearities (α = 0.01). This is followed by a fully connected layer of size 64 and then a 
    scalar output. The agent action at is not used as input as this did not improve performance. 

    - Convolutional layers use batch normalization (Ioffe and Szegedy, 2015) 
    - TODO: with decay rate 0.99
    - Conv layers have per-channel dropout (Srivastava et al., 2014) with α = 0.8.
    """

    def __init__(self):
        super().__init__()
        # pov feature extractor
        # 64x64
        channel = 16
        relu_negative_slope = 0.01
        dropout_rate = 0.2
        conv_out_size = 64
        # TODO: vec_size == vec.shape[1] (This needs to adapted for each environment)
        vec_size = 27 # findCave
        self.conv = nn.Sequential(
            nn.Conv2d(3, channel, 7, stride=3),
            nn.BatchNorm2d(channel),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(negative_slope=relu_negative_slope, inplace=True),

            nn.Conv2d(channel, channel, 5, stride=2),
            nn.BatchNorm2d(channel),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(negative_slope=relu_negative_slope, inplace=True),
            
            nn.Conv2d(channel, channel, 3, stride=1),
            nn.BatchNorm2d(channel),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(negative_slope=relu_negative_slope, inplace=True),
            
            nn.Conv2d(channel, channel, 3, stride=1),
            nn.BatchNorm2d(channel),
            nn.Dropout(dropout_rate),
            nn.LeakyReLU(negative_slope=relu_negative_slope, inplace=True),
            
            nn.Flatten(),
            nn.Linear(256, conv_out_size),
            # nn.Linear(64, 1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_size + vec_size, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1),
        )

    def forward(self, pov, vec):
        hid = self.conv(pov)
        out = self.mlp(
            torch.cat([hid, vec], dim=1)
        )
        return out
