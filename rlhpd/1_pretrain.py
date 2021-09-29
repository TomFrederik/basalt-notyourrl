"""
Step 2 in the algorithm:
Pretrain the policy with expert demos
"""

import argparse
from pathlib import Path

from common import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run behavioral cloning pretraining with expert demos')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = utils.load_config(options.config_file)

    model_path = Path(cfg.pretrain_policy.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Running behavioral cloning pretraining!")
    print("Saving model to", model_path)
    with open(model_path, "w") as f:
        f.write("DUMMY MODEL")