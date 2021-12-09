"""
Step 2 in the algorithm:
Pretrain the policy with expert demos
"""

import argparse
from pathlib import Path

from common import utils

from DQfD_pretraining import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run behavioral cloning pretraining with expert demos')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = utils.load_config(options.config_file)
    
    print("Running DQfD pretraining!")
    args = {**vars(cfg.pretrain_dqfd_args), **vars(cfg.dqfd_args)} # join common dqfd args with those that are specific for pretraining
    main(**args)