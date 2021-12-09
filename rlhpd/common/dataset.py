import argparse
import pickle
from pathlib import Path

import einops
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from . import preference_helpers as pref
from . import utils
from .database import AnnotationBuffer

class TrajectoryPreferencesDataset(Dataset):
    """Trajectory preferences dataset."""

    def __init__(self, annotation_db_path=None):
        """
        Args:
            data_dir (string): Path to the directory containing pickled trajectory
                (state, action, reward) tuples
            annotation_db_path (string): Path to the annotation db
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotation_db_path = Path(annotation_db_path)
        assert self.annotation_db_path.is_file()

        db = AnnotationBuffer(annotation_db_path)
        self.traj_path_pairs, labels = db.get_all_rated_pairs_with_labels()
        self.labels = [db.label_to_judgement(l) for l in labels]

    def __len__(self):
        return len(self.traj_path_pairs)

    def __getitem__(self, idx):
        """
        Loads a single sample of data
        """
        # Load from file
        traj_path_a, traj_path_b = self.traj_path_pairs[idx]
        clip_a = utils.load_clip_from_file(traj_path_a)
        clip_b = utils.load_clip_from_file(traj_path_b)
        assert len(clip_a) == len(clip_b), (traj_path_a, traj_path_b)

        # if self.labels is None:
        #     # Compute judgement based on rewards
        #     judgement = torch.as_tensor(pref.simulate_judgement(clip_a, clip_b))
        # else:
        judgement = torch.as_tensor(self.labels[idx])

        frames_a, vec_a = utils.get_frames_and_vec_from_clip(clip_a)
        frames_b, vec_b = utils.get_frames_and_vec_from_clip(clip_b)
        # vec_a[:] = 0
        # vec_b[:] = 0
        # frames_a[:] = 0
        # frames_b[:] = 0
        # print(vec_a)
        # print(vec_b)
        # print(frames_a)
        # print(frames_b)

        # We need to occasionally swap the positions of A and B clips because
        # autolabeling always puts the better clip on the left, so without this
        # swapping, the model will learn to simply "always predict left"
        # Edit: This should not in fact be true since the model doesn't receive
        # but only single images without context, but will leave this in until we
        # verify what the actual problem is
        if idx % 2 == 0:
            frames_a, vec_a, frames_b, vec_b = frames_b, vec_b, frames_a, vec_a
            judgement = torch.tensor([1,1]) - judgement

        assert frames_a.shape == frames_b.shape
        assert frames_a.shape == (len(frames_a), 3, 64, 64)
        assert judgement.shape == (2,)
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
