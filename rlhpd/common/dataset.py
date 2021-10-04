import argparse
import pickle
from pathlib import Path

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import preference_helpers as pref


class TrajectoryPreferencesDataset(Dataset):
    """Trajectory preferences dataset."""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): Path to the directory containing pickled trajectory
                (state, action, reward) tuples
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        traj_paths = sorted([x for x in Path(data_dir).glob("*.pickle")])
        # Populate all possible pairs
        path_pairs = []
        for path_a in traj_paths:
            for path_b in traj_paths:
                if path_a == path_b:
                    continue
                path_pairs.append((path_a, path_b))
        self.traj_path_pairs = path_pairs
        self.transform = transform

    def __len__(self):
        return len(self.traj_path_pairs)

    def __getitem__(self, idx):
        """
        Loads a batch of data
        """
        # Load from file
        traj_path_a, traj_path_b = self.traj_path_pairs[idx]
        with open(traj_path_a, 'rb') as f:
            clip_a = pickle.load(f)
        with open(traj_path_b, 'rb') as f:
            clip_b = pickle.load(f)
        assert len(clip_a) == len(clip_b)

        # Compute judgement based on rewards
        judgement = torch.as_tensor(pref.simulate_judgement(clip_a, clip_b))

        # Preprocess images
        frames_a = torch.stack([torch.as_tensor(state['pov'], dtype=torch.float32) for (state, action, reward, next_state, done, meta) in clip_a], axis=0)
        frames_a = einops.rearrange(frames_a, 't h w c -> t c h w') / 255
        vec_a = torch.stack([torch.as_tensor(state['vector'], dtype=torch.float32) for (state, action, reward, next_state, done, meta) in clip_a], axis=0)
        frames_b = torch.stack([torch.as_tensor(state['pov'], dtype=torch.float32) for (state, action, reward, next_state, done, meta) in clip_b], axis=0)
        frames_b = einops.rearrange(frames_b, 't h w c -> t c h w') / 255
        vec_b = torch.stack([torch.as_tensor(state['vector'], dtype=torch.float32) for (state, action, reward, next_state, done, meta) in clip_a], axis=0)

        sample = {
            # State a
            'frames_a': frames_a,
            'vec_a': vec_a,
            # State b
            'frames_b': frames_b,
            'vec_b': vec_b,
            # Preference
            'judgement': judgement,
        }
        return sample


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model')
    parser.add_argument("-c", "--clips-dir", default="./output",
                        help="Clips directory. Default: %(default)s")
    options = parser.parse_args()

    full_dataset = TrajectoryPreferencesDataset(options.clips_dir)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    for i_batch, sample_batched in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        frames_a = sample_batched['frames_a']
        frames_b = sample_batched['frames_b']
        vec_a = sample_batched['vec_a']
        vec_b = sample_batched['vec_b']
        judgements = sample_batched['judgement']
        assert len(frames_a) == len(judgements)

    for i_batch, sample_batched in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
        frames_a = sample_batched['frames_a']
        frames_b = sample_batched['frames_b']
        judgements = sample_batched['judgement']
        vec_a = sample_batched['vec_a']
        vec_b = sample_batched['vec_b']
        assert len(frames_a) == len(judgements)
