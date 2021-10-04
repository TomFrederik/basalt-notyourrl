import random

import munch
import numpy as np
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
