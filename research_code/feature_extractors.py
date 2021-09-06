import gym
import torch
import torch.nn as nn
from vqvae import VQVAE
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import einops

class VQVAEFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, vqvae_path, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # load vqvae
        self.vqvae = VQVAE.load_from_checkpoint(vqvae_path)
        self.vqvae.eval()
        
        # conv net that operates on the vqvae latent image and distills it down to a single vector
        # TODO make conv net customizable via kwargs
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=self.vqvae.hparams.embedding_dim, out_channels=256, kernel_size=3, padding=1, stride=2), # 16 -> 8
            nn.GELU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=2), # 8 -> 4
            nn.GELU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=2), # 4 -> 2
            nn.GELU(),
            nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1, stride=2), # 2 -> 1
            nn.GELU()
        )
        
        # linear net to map the output of the conv net to the correct dimensionality
        self.linear = nn.Sequential(
            nn.Linear(2048, features_dim),
            nn.GELU()
        )
        
    def forward(self, observations):
        out = self.vqvae.encode_only(observations)[0]
        out = self.conv_net(out)
        out = self.linear(einops.rearrange(out, 'b c h w -> b (c h w)')) # flatten image from (C 1 1) -> C
        return out
    