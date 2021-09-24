from os import access
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

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
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import utils
import dataset

def pred_pref_probs(reward_model, frames_a, frames_b, judgements):
    batch_imgs = torch.stack([frames_a, frames_b], axis=1)
    current_batch_size = len(batch_imgs) # == cfg.batch_size except at the end of the dataset
    assert batch_imgs.shape == (current_batch_size, 2, cfg.clip_length, 3, 64, 64)
    assert judgements.shape == (current_batch_size, 2)
    
    # Flatten the (batch, Trajectory, time) dimensions to feed B,C,W,H into CNN
    batch_imgs = einops.rearrange(batch_imgs, 'b T t c w h -> (b T t) c w h')
    assert batch_imgs.shape == (current_batch_size * 2 * cfg.clip_length, 3, 64, 64)

    # Predict reward of every frame in our batch
    r_preds = reward_model(torch.as_tensor(batch_imgs, dtype=torch.float32)).squeeze()
    assert r_preds.shape == (len(batch_imgs),)

    # Reshape and sum up rewards along each
    all_rewards = r_preds.view(current_batch_size, 2, cfg.clip_length)
    assert all_rewards.shape == (current_batch_size, 2, cfg.clip_length)
    reward_sums = torch.sum(all_rewards, dim=2)
    assert reward_sums.shape == (current_batch_size, 2)

    # Convert pairs of rewards into preference probabilities
    prefer_probs = torch.softmax(reward_sums, dim=1)
    assert prefer_probs.shape == (current_batch_size, 2)
    return prefer_probs

def get_pred_accuracy(prefer_probs, true_judgements):
    pred_judgements = utils.probs_to_judgements(prefer_probs.detach().numpy())
    assert pred_judgements.shape == true_judgements.shape
    eq = np.all(true_judgements.numpy() == pred_judgements, axis=1)
    acc = eq.sum() / len(eq)
    return acc

def set_seeds(num):
    torch.manual_seed(num)
    random.seed(num)
    np.random.seed(num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model')
    parser.add_argument("-c", "--clips-dir", default="./output",
                        help="Clips directory. Default: %(default)s")
    options = parser.parse_args()

    wandb.init(project='DRLfHP-cartpole', entity='junshern')
    # wandb.init(project='DRLfHP-cartpole', entity='junshern', mode="disabled")

    # TODO: The model is trained on batches of 16 segment pairs (see below), 
    # optimized with Adam (Kingma and Ba, 2014) 
    # with learning rate 0.0003, β1 = 0.9, β2 = 0.999, and  = 10−8 .
    cfg = wandb.config
    cfg.learning_rate = 1e-4
    cfg.batch_size = 16
    cfg.max_num_pairs = None
    cfg.rand_seed = 0
    cfg.val_split = 0.1
    cfg.log_every = 50

    set_seeds(cfg.rand_seed)

    reward_model = RewardModel()
    wandb.watch(reward_model)
    optimizer = torch.optim.SGD(reward_model.parameters(), lr=cfg.learning_rate)

    traj_paths = sorted([x for x in Path(options.clips_dir).glob("*.pickle")])

    # Check the length of a single clip
    with open(traj_paths[0], 'rb') as f:
        clip = pickle.load(f)
        cfg.clip_length = len(clip)

    # resize_imgs = transforms.Compose([transforms.Resize((64, 64))])
    full_dataset = dataset.TrajectoryPreferencesDataset(options.clips_dir)
    val_size = int(cfg.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    for batch_idx, data_batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        frames_a, frames_b, judgements = \
            data_batch['frames_a'], data_batch['frames_b'], data_batch['judgement']

        # Run prediction pipeline
        prefer_probs = pred_pref_probs(reward_model, frames_a, frames_b, judgements)

        # Calculate loss:
        # This is ALMOST CrossEntropy but slightly different because we want to support
        # tie condition (0.5, 0.5) in judgement which is not possible in default XEnt
        # loss = - (judgement[0] * torch.log(prefer_1) + judgement[1] * torch.log(prefer_2))
        assert judgements.shape == prefer_probs.shape
        loss = - torch.sum(judgements * torch.log(prefer_probs))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation & logging
        if batch_idx % cfg.log_every == 0:
            with torch.no_grad():
                # Train accuracy
                train_acc = get_pred_accuracy(prefer_probs, judgements)
                # Validation accuracy
                for val_batch_idx, val_batch in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
                    val_frames_a, val_frames_b, val_judgements = \
                        val_batch['frames_a'], val_batch['frames_b'], val_batch['judgement']
                    val_probs = pred_pref_probs(
                        reward_model, val_frames_a, val_frames_b, val_judgements)
                    val_acc = get_pred_accuracy(val_probs, val_judgements)
                # Log metrics
                wandb.log({"loss": loss.item()})
                wandb.log({"train_acc": train_acc})
                wandb.log({"val_acc": val_acc})

    #     # TODO: A fraction of 1/e of the data is held out to be used as a validation set. 
    #     # We use L2- regularization of network weights with the adaptive scheme described in 
    #     # Christiano et al. (2017): the L2-regularization weight increases if the average 
    #     # validation loss is more than 50% higher than the average training loss, and decreases 
    #     # if it is less than 10% higher (initial weight 0.0001, multiplicative rate of change 
    #     # 0.001 per learning step).

    #     # TODO: An extra loss proportional to the square of the predicted rewards is added 
    #     # to impose a zero-mean Gaussian prior on the reward distribution.

    #     # TODO: Gaussian noise of amplitude 0.1 (the grayscale range is 0 to 1) is added to the inputs.
    #     # TODO: Grayscale images?

    #     # TODO: We assume there is a 10% chance that the annotator responds uniformly at random, 
    #     # so that the model will not overfit to possibly erroneous preferences. We account for 
    #     # this error rate by using Pˆ e = 0.9Pˆ + 0.05 instead of Pˆ for the cross-entropy computation.

    #     # TODO: since the reward model is trained merely on comparisons, its absolute scale is arbitrary. 
    #     # Therefore we normalize its output so that it has 0 mean and standard deviation 0.05 
    #     # across the annotation buffer
    #     # JS: This is only when passing as output to the RL algorithm?
    #     # See https://github.com/mrahtz/learning-from-human-preferences/blob/master/reward_predictor.py#L167-L169

    #         # print("Running behavioral cloning pretraining!")
    #         # model_path = Path(cfg["out_models_dir"]) / "Q_0.pth"
    #         # model_path.parent.mkdir(parents=True, exist_ok=True)
    #         # print("Saving model to", model_path)
    #         # with open(model_path, "w") as f:
    #         #     f.write("DUMMY MODEL")