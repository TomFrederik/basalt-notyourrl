"""
Steps 3 & 4 in the algorithm:
Run pretrained model in environment to get trajectories
Sample clips from trajectories
Add clips (unannotated) into annotation db
"""

import argparse
from pathlib import Path

import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample clips from pretrained model')
    parser.add_argument("-r", "--run-id", required=True, help="ID for current run")
    parser.add_argument("-o", "--output-basedir", default="./output", help="Base dir for outputs")
    options = parser.parse_args()

    cfg = config.load_config(Path(options.output_basedir) / options.run_id / "run.yaml")

    model_path = Path(cfg["out_models_dir"]) / "Q_0.pth"
    assert model_path.exists()
    print("Loading pretrained model from", model_path)
    print("Running model in", cfg['env_task'])
    print(f"Sampling {cfg['clip_sampler']['num_clips']} clips of length {cfg['clip_sampler']['clip_length']}")
    print("Adding pairs of sampled clips to", cfg['out_annotation_db'])
    with open(cfg['out_annotation_db'], 'w') as f:
        f.write("DUMMY DB")