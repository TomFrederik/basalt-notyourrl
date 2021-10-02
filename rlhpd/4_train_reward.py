"""
Step 7 in the algorithm:
Train reward model with preferences from the annotation buffer
"""

import argparse
from pathlib import Path

from common import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model with initial preferences')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = utils.load_config(options.config_file)