import argparse
import pickle
import uuid
from pathlib import Path

import minerl
import common.utils as utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate clips from human demos')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    # Params
    cfg = utils.load_config(options.config_file)
    clips_dir = Path(cfg.clips.dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_length = cfg.clips.clip_length
    num_clips = cfg.clips.num_clips

    minerl_data = minerl.data.make(cfg.env_task, data_dir=cfg.demos_dir)
    traj_names = minerl_data.get_trajectory_names()

    clips_counter = 0
    for traj_name in traj_names:
        data_frames = list(minerl_data.load_data(traj_name, include_metadata=True))
        start_idxs = list(range(len(data_frames)))[-clip_length::-clip_length]
        for start_idx in start_idxs:
            clip = data_frames[start_idx: start_idx + clip_length]
            # clip == list of (state, action, reward, next_state, done, meta)
            
            # Save pickle
            unique_filename = str(uuid.uuid4())
            outfile = (clips_dir / unique_filename).with_suffix('.pickle')
            with open(outfile, 'wb') as f:
                pickle.dump(clip, f)
            
            clips_counter += 1
            if clips_counter >= num_clips:
                break
        if clips_counter >= num_clips:
            break