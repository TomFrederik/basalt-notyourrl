"""
Step 8 in the algorithm:
Train DQfD with expert demos and initial reward model

IMPORTANT: Make sure to execute this script with 'xvfb-run -a python ...' to enable the minerl gym environment
"""

import argparse
from pathlib import Path

from common import utils

from DQfD_training import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQfD with expert demos and initial reward model')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = utils.load_config(options.config_file)
    
    print("Running DQfD training!")
    args = {**vars(cfg.train_dqfd_args), **vars(cfg.dqfd_args)} # join common dqfd args with those that are specific for training
    main(**args)