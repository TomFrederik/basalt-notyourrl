"""
Step 8 in the algorithm:
Train DQfD with expert demos and initial reward model
"""

import argparse
from pathlib import Path

from common import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQfD with expert demos and initial reward model')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = utils.load_config(options.config_file)