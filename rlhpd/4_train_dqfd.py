"""
Step 8 in the algorithm:
Train DQfD with expert demos and initial reward model
"""

import argparse
from pathlib import Path

import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQfD with expert demos and initial reward model')
    parser.add_argument("-r", "--run-id", required=True, help="ID for current run")
    parser.add_argument("-o", "--output-basedir", default="./output", help="Base dir for outputs")
    options = parser.parse_args()

    cfg = config.load_config(Path(options.output_basedir) / options.run_id / "run.yaml")

    policy_model_path = Path(cfg["out_models_dir"]) / "Q_0.pth"
    reward_model_path = Path(cfg["out_models_dir"]) / "Reward_1.pth"
    assert policy_model_path.exists()
    assert reward_model_path.exists()
    
    print("Training DQfD...")
    new_policy_model_path = Path(cfg["out_models_dir"]) / "Q_1.pth"
    new_policy_model_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving model to", new_policy_model_path)
    with open(new_policy_model_path, "w") as f:
        f.write("DUMMY MODEL")