import pickle
import random

import einops
import munch
import numpy as np
import skvideo.io
import torch
import yaml


def set_seeds(num):
    torch.manual_seed(num)
    random.seed(num)
    np.random.seed(num)

def load_config(config_path: str) -> munch.Munch:
    with open(config_path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        cfg = munch.munchify(cfg_dict)
    return cfg

def load_clip_from_file(filepath):
    with open(filepath, 'rb') as f:
        clip = pickle.load(f)
    return clip

def get_frames_and_vec_from_clip(clip: list):
    """
    clip: list of tuples (state, action, reward, next_state, done, meta)
    """
    frames = torch.stack([torch.as_tensor(state['pov'], dtype=torch.float32) for (state, action, reward, next_state, done, meta) in clip], axis=0)
    vec = torch.stack([torch.as_tensor(state['vec'], dtype=torch.float32) for (state, action, reward, next_state, done, meta) in clip], axis=0)
    return frames, vec

def pov_obs_to_img(pov):
    """
    Reverses the state shaping: convert from NN-friendly to visualize-friendly
    """
    return (einops.rearrange(pov, 'c h w -> h w c') * 255).astype(np.uint8)

def save_vid(imgs, video_path, fps):
    assert len(imgs) > 0
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with skvideo.io.FFmpegWriter(
        video_path, 
        inputdict={'-r': str(fps)},
        outputdict={'-r': str(fps), '-vcodec': 'libx264'},
        ) as writer:
        for idx in range(imgs.shape[0]):
            img = pov_obs_to_img(imgs[idx])
            writer.writeFrame(img)
