import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Linear(256, 64),
            nn.Linear(64, 1),
        )

    def forward(self, obs):
        out = self.conv(obs)
        return out


import argparse
import numpy as np
import pickle
from pathlib import Path

import einops
import torch
from torchvision import transforms
from tqdm import tqdm

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model')
    parser.add_argument("-c", "--clips-dir", default="./output",
                        help="Clips directory. Default: %(default)s")
    options = parser.parse_args()

    traj_paths = sorted([x for x in Path(options.clips_dir).glob("*.pickle")])

    learning_rate = 1e-4

    reward_model = RewardModel()
    optimizer = torch.optim.SGD(reward_model.parameters(), lr=learning_rate)

    resize_imgs = transforms.Compose([transforms.Resize((64, 64))])

    for i in tqdm(range(len(traj_paths))):
        for j in range(len(traj_paths)):
            if i == j:
                continue
            with open(traj_paths[i], 'rb') as f:
                clip_1 = pickle.load(f)
            with open(traj_paths[j], 'rb') as f:
                clip_2 = pickle.load(f)

            judgement = utils.simulate_judgement(clip_1, clip_2)

            # Preprocess images
            imgs_1 = np.stack([img for (img, action, reward) in clip_1], axis=0)
            imgs_2 = np.stack([img for (img, action, reward) in clip_2], axis=0)
            # Reshape and scale
            imgs_1 = einops.rearrange(imgs_1, 'b h w c -> b c h w') / 255
            imgs_2 = einops.rearrange(imgs_2, 'b h w c -> b c h w') / 255
            imgs_1 = resize_imgs(torch.as_tensor(imgs_1, dtype=torch.float32))
            imgs_2 = resize_imgs(torch.as_tensor(imgs_2, dtype=torch.float32))
            assert imgs_1.shape == (len(imgs_1), 3, 64, 64)

            out_1 = reward_model(imgs_1)
            out_2 = reward_model(imgs_2)
            assert out_1.shape == (len(imgs_1), 1)
            sum_rewards_1 = torch.sum(out_1)
            sum_rewards_2 = torch.sum(out_2)
            print(out_1)
            print(out_2)
            print(sum_rewards_1, sum_rewards_2)
            prefer_1 = torch.exp(sum_rewards_1) / (torch.exp(sum_rewards_1) + torch.exp(sum_rewards_2))
            prefer_2 = torch.exp(sum_rewards_2) / (torch.exp(sum_rewards_1) + torch.exp(sum_rewards_2))
            print(prefer_1, prefer_2)
            loss = - judgement[0] * torch.log(prefer_1) + judgement[1] * torch.log(prefer_2)
            print(loss)

            # TODO: Should I sum up all losses then make one big gradient update,
            # or make a gradient update on each loss?

            # TODO: A fraction of 1/e of the data is held out to be used as a validation set. 
            # We use L2- regularization of network weights with the adaptive scheme described in 
            # Christiano et al. (2017): the L2-regularization weight increases if the average 
            # validation loss is more than 50% higher than the average training loss, and decreases 
            # if it is less than 10% higher (initial weight 0.0001, multiplicative rate of change 
            # 0.001 per learning step).

            # TODO: An extra loss proportional to the square of the predicted rewards is added 
            # to impose a zero-mean Gaussian prior on the reward distribution.

            # TODO: Gaussian noise of amplitude 0.1 (the grayscale range is 0 to 1) is added to the inputs.
            # TODO: Grayscale images?

            # TODO: We assume there is a 10% chance that the annotator responds uniformly at random, 
            # so that the model will not overfit to possibly erroneous preferences. We account for 
            # this error rate by using Pˆ e = 0.9Pˆ + 0.05 instead of Pˆ for the cross-entropy computation.

            # TODO: since the reward model is trained merely on comparisons, its absolute scale is arbitrary. 
            # Therefore we normalize its output so that it has 0 mean and standard deviation 0.05 
            # across the annotation buffer

            # TODO: The model is trained on batches of 16 segment pairs (see below), 
            # optimized with Adam (Kingma and Ba, 2014) 
            # with learning rate 0.0003, β1 = 0.9, β2 = 0.999, and  = 10−8 .

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"loss: {loss.item():>7f}")
            
        #     if j > 1:
        #         break
        # break

            # print("Running behavioral cloning pretraining!")
            # model_path = Path(cfg["out_models_dir"]) / "Q_0.pth"
            # model_path.parent.mkdir(parents=True, exist_ok=True)
            # print("Saving model to", model_path)
            # with open(model_path, "w") as f:
            #     f.write("DUMMY MODEL")