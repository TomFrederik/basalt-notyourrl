from pathlib import Path
import config
import DQfD
# import runner

CONFIG_FILE = "config.yaml"

if __name__ == '__main__':
    cfg = config.initialize(CONFIG_FILE)

    print("Running pretraining!")

    print("Saving model to", Path(cfg["out_models_dir"]) / "Q_pre.pth")