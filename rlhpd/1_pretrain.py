"""
Step 2 in the algorithm:
Pretrain the policy with expert demos
"""

import argparse
from pathlib import Path

from common import utils

# TODO: adapt when changing location of DQfD
from research_code.Pretrain_DQfD import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run behavioral cloning pretraining with expert demos')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = utils.load_config(options.config_file)

    model_path = Path(cfg.pretrain_dqfd_args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Running DQfD pretraining!")
    main(**vars(cfg.pretrain_dqfd_args))