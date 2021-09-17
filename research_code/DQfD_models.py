import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class QNetwork(nn.Module):
    def __init__(self, num_actions, inv_dim, inv_network_dim=128, inv_feature_dim=128, n_hid=64, pov_feature_dim=128, q_net_dim=128):
        super().__init__()
        # pov feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(3, n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_hid, 2*n_hid, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*n_hid, 2*n_hid, 3, padding=1),
            nn.ReLU(inplace=True),
            ResBlock(2*n_hid, 2*n_hid//4),
            ResBlock(2*n_hid, 2*n_hid//4),
            Rearrange('b c h w -> (b h w) c'),
            nn.Linear(2*n_hid, pov_feature_dim)
        )
        
        # feature extractor for other observations
        # expects that those other observations are already stacked into a float tensor
        self.inv_network = nn.Sequential(
            nn.Linear(inv_dim, inv_network_dim),
            nn.ReLU(inplace=True),
            nn.Linear(inv_network_dim, inv_network_dim),
            nn.ReLU(inplace=True),
            nn.Linear(inv_network_dim, inv_feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # final q network
        self.q_net = nn.Sequential(
            nn.Linear(pov_feature_dim + inv_feature_dim, q_net_dim),
            nn.ReLU(inplace=True),
            nn.Linear(q_net_dim, q_net_dim),
            nn.ReLU(inplace=True),
            nn.Linear(q_net_dim, num_actions),
        )
    
    def forward(self, obs):
        # apply conv net to pov_obs
        pov_features = self.conv(obs['pov'])
        
        # preprocess other observations
        inv_features = self.inv_network(obs['inv'])
        
        # concat inputs
        q_net_input = torch.cat([pov_features, inv_features], dim=1)
        
        return self.q_net(q_net_input)
    

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out
