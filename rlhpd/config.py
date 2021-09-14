from datetime import datetime
from pathlib import Path

import yaml

def initialize(cfg_file):
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    # Create new ID based on timestamp to identify our run
    if cfg["run_id"] is None:
        timestr = datetime.utcnow().strftime('run-%Y%m%d-%H%M%S.%f')[:-3]
        cfg["run_id"] = timestr
    # Each output_dir is created 
    output_dir = Path(cfg["outputs_base"]) / cfg["run_id"]
    # Set output paths relative to output_dir
    for key in cfg:
        if key.startswith("out_"):
            cfg[key] = str(output_dir / cfg[key])
    print(f"Initializing run (save this ID for subsequent runs): {cfg['run_id']}")
    
    out_cfg_file = Path(cfg["out_cfg_file"])
    out_cfg_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cfg_file, "w") as f:
        yaml.dump(cfg, f)
    return cfg

def load_config(cfg_file):
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def print_config(cfg):
    print(yaml.safe_dump(cfg, allow_unicode=True, default_flow_style=False))