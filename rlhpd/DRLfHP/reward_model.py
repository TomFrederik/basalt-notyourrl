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
import random
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

    # TODO: The model is trained on batches of 16 segment pairs (see below), 
    # optimized with Adam (Kingma and Ba, 2014) 
    # with learning rate 0.0003, β1 = 0.9, β2 = 0.999, and  = 10−8 .
    learning_rate = 1e-4
    batch_size = 16
    max_num_pairs = None
    rand_seed = 0

    reward_model = RewardModel()
    optimizer = torch.optim.SGD(reward_model.parameters(), lr=learning_rate)

    resize_imgs = transforms.Compose([transforms.Resize((64, 64))])

    # Populate all possible pairs
    traj_path_pairs = []
    for path_a in traj_paths:
        for path_b in traj_paths:
            if path_a == path_b:
                continue
            traj_path_pairs.append((path_a, path_b))
    random.seed(rand_seed)
    random.shuffle(traj_path_pairs)

    if max_num_pairs is not None:
        traj_path_pairs = traj_path_pairs[:max_num_pairs]
    # Iterate through pairs and process
    pair_idx = 0
    num_batches = int(np.ceil(len(traj_path_pairs) / batch_size))
    losses = []
    for batch_idx in tqdm(range(num_batches)):
        # Load one batch of data
        batch_clip_frames = []  # List of frames (batch_size * 2 * clip_length)
        batch_judgements = []   # List of per-clip judgements (batch_size * 2)
        for current_batch_size in range(1, batch_size + 1):
            traj_path_a, traj_path_b = traj_path_pairs[pair_idx]
            with open(traj_path_a, 'rb') as f:
                clip_a = pickle.load(f)
            with open(traj_path_b, 'rb') as f:
                clip_b = pickle.load(f)
            assert len(clip_a) == len(clip_b)
            clip_length = len(clip_a)
            # Store one flat list of [*clip_a_1, *clip_b_1, *clip_a_2, ...]
            batch_clip_frames += clip_a + clip_b
            batch_judgements.append(utils.simulate_judgement(clip_a, clip_b))
            pair_idx += 1

            if pair_idx >= len(traj_path_pairs):
                break

        # Preprocess images
        imgs = np.stack([img for (img, action, reward) in batch_clip_frames], axis=0)
        assert imgs.shape == (len(batch_clip_frames), 64, 64, 3)
        # Reshape and scale
        imgs = einops.rearrange(imgs, 'b h w c -> b c h w') / 255
        # imgs = resize_imgs(torch.as_tensor(imgs, dtype=torch.float32))
        assert imgs.shape == (len(batch_clip_frames), 3, 64, 64)

        # Predict reward of every frame in our batch
        r_preds = reward_model(torch.as_tensor(imgs, dtype=torch.float32)).squeeze()
        assert r_preds.shape == (len(batch_clip_frames),)

        # Reshape and sum up rewards along each
        all_rewards = r_preds.view(current_batch_size, 2, clip_length)
        assert all_rewards.shape == (current_batch_size, 2, clip_length)
        reward_sums = torch.sum(all_rewards, dim=2)
        assert reward_sums.shape == (current_batch_size, 2)

        # Convert pairs of rewards into preference probabilities
        prefer_probs = torch.softmax(reward_sums, dim=1)
        assert prefer_probs.shape == (current_batch_size, 2)

        # Calculate loss:
        # This is ALMOST CrossEntropy but slightly different because we want to support
        # tie condition (0.5, 0.5) in judgement which is not possible in default XEnt
        # loss = - (judgement[0] * torch.log(prefer_1) + judgement[1] * torch.log(prefer_2))
        batch_judgements = torch.as_tensor(batch_judgements)
        assert batch_judgements.shape == prefer_probs.shape
        loss = - torch.sum(batch_judgements * torch.log(prefer_probs))
        # print(f"loss: {loss.item():>7f}")
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        # JS: This is only when passing as output to the RL algorithm?
        # See https://github.com/mrahtz/learning-from-human-preferences/blob/master/reward_predictor.py#L167-L169


            # print("Running behavioral cloning pretraining!")
            # model_path = Path(cfg["out_models_dir"]) / "Q_0.pth"
            # model_path.parent.mkdir(parents=True, exist_ok=True)
            # print("Saving model to", model_path)
            # with open(model_path, "w") as f:
            #     f.write("DUMMY MODEL")
    print(losses)