"""
Steps 3 & 4 in the algorithm:
Run pretrained model in environment to get trajectories
Sample clips from trajectories
Add clips (unannotated) into annotation db
"""

import argparse
import os
import random
import time
from pathlib import Path

import gym
import minerl  # This is required to be able to import minerl environments
import numpy as np
from tqdm import tqdm

from common import database, utils


def step_random_policy(environment):
    action = environment.action_space.sample()
    observation, _, done, _ = environment.step(action)
    return observation, done

def generate_trajectory(max_traj_length,environment):
    done = False
    observation = environment.reset()

    trajectory = np.zeros(shape=(max_traj_length, 64, 64, 3))
    step_idx = 0
    while not done and step_idx < max_traj_length:
        trajectory[step_idx] = observation["pov"]
        observation, done = step_random_policy(environment)
        step_idx += 1
        environment.render()
    return trajectory

def generate_sample(trajectory, max_traj_length, sample_length):
    assert max_traj_length > sample_length
    starting_idx = rng.integers(low=0, high=max_traj_length-sample_length)
    sample = trajectory[starting_idx:starting_idx+sample_length,...]
    return sample

def fill_database(db, num_of_traj, num_of_samples, run_id, pair_per_sample):
    #names will be traj_x_smpl_y
    existing_ids = db.return_all_ids()
    if existing_ids:
        for x in range(num_of_traj):
            for y in range(num_of_samples):
                for _ in range(pair_per_sample):
                    random_match = random.choice(existing_ids)[0]
                    # print(random_match, run_id, x, y)
                    try: # if we picked a pair that exists we just skip for now TODO
                        db.insert_traj_pair(
                            f"{run_id}_traj_{x}_smpl_{y}", random_match)
                    except:
                        continue

    else: # no samples in database so far, add each id once to get started 
        for x1 in range(num_of_traj):
            x2 = np.mod(x1+1, num_of_traj)
            for y in range(num_of_samples):
                db.insert_traj_pair(
                    f"{run_id}_traj_{x1}_smpl_{y}",
                    f"{run_id}_traj_{x2}_smpl_{y}"
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample clips from pretrained model')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    # Load config params
    cfg = utils.load_config(options.config_file)
    rng = np.random.default_rng(cfg.sampler.rnd_seed)
    env_task = cfg.env_task
    num_traj = cfg.sampler.num_traj
    max_traj_length = cfg.sampler.max_traj_length
    num_samples = cfg.sampler.num_samples
    sample_length = cfg.sampler.sample_length
    pair_per_sample = cfg.sampler.pair_per_sample
    db_path = Path(cfg.sampler.db_path)
    traj_dir = Path(cfg.sampler.traj_dir)

    print(f"Initializing environment {env_task}. This might take a while...")
    environment = gym.make(env_task)
    print("Done initializing environment!")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = database.AnnotationBuffer(db_path)
    os.makedirs(traj_dir, exist_ok=True)
    run_id = time.strftime('%Y%m%d-%H%M%S')
    
    for traj_idx in tqdm(range(num_traj)):
        trajectory = generate_trajectory(max_traj_length, environment)
        traj_path = traj_dir / f"{run_id}_traj_{traj_idx}_full"
        np.save(traj_path, trajectory)

        for sample_idx in range(num_samples):
            sample = generate_sample(trajectory, max_traj_length, sample_length)
            sample_path = traj_dir / f"{run_id}_traj_{traj_idx}_smpl_{sample_idx}"
            np.save(sample_path, sample)
            fill_database(db, num_traj, num_samples, run_id, pair_per_sample)
