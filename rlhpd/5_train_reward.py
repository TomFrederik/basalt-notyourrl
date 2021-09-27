"""
Step 7 in the algorithm:
Train reward model with preferences from the annotation buffer
"""

import argparse
from pathlib import Path

import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model with initial preferences')
    parser.add_argument("-r", "--run-id", required=True, help="ID for current run")
    parser.add_argument("-o", "--output-basedir", default="./output", help="Base dir for outputs")
    options = parser.parse_args()

    cfg = config.load_config(Path(options.output_basedir) / options.run_id / "run.yaml")

    annotation_db_path = Path(cfg['out_annotation_db'])
    print("Getting preference pairs from", annotation_db_path)
    assert annotation_db_path.exists()
    print("Training reward model...")
    model_path = Path(cfg["out_models_dir"]) / "Reward_1.pth"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print("Saving model to", model_path)
    with open(model_path, "w") as f:
        f.write("DUMMY MODEL")