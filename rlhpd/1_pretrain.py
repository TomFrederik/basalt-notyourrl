"""
Step 2 in the algorithm:
Pretrain the policy with expert demos
"""

import argparse
from pathlib import Path

import config
import DQfD

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run behavioral cloning pretraining with expert demos')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = config.initialize(options.config_file)

    print("Running behavioral cloning pretraining!")
    model_path = Path(cfg["out_models_dir"]) / "Q_0.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving model to", model_path)
    with open(model_path, "w") as f:
        f.write("DUMMY MODEL")