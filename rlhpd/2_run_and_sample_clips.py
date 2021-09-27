"""
Steps 3 & 4 in the algorithm:
Run pretrained model in environment to get trajectories
Sample clips from trajectories
Add clips (unannotated) into annotation db
"""

import argparse
import os
import random
from pathlib import Path

import gym
import minerl # This is required to be able to import minerl environments
import numpy as np
from tqdm import tqdm

import config
import database


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

def save_trajectory(traj_name, trajectory, random_run_num):
    np.save(f"trajectories/{random_run_num}_traj_{traj_name}_full", trajectory)

def generate_sample(trajectory, max_traj_length, sample_length):
    assert max_traj_length > sample_length
    starting_idx =rng.integers(low=0, high=max_traj_length-sample_length)
    sample = trajectory[starting_idx:starting_idx+sample_length,...]
    return sample

def save_sample(traj_name,sample_name, sample, random_run_num):
    np.save(f"trajectories/{random_run_num}_traj_{traj_name}_smpl_{sample_name}", sample)

def fill_database(db, num_of_traj, num_of_samples, random_run_num, pair_per_sample):
    #names will be traj_x_smpl_y
    existing_ids = db.return_all_ids()
    if existing_ids:
        for x in range(num_of_traj):
            for y in range(num_of_samples):
                for _ in range(pair_per_sample):
                    random_match = random.choice(existing_ids)[0]
                    print(random_match, random_run_num, x, y)
                    try: # if we picked a pair that exists we just skip for now TODO
                        db.insert_traj_pair(
                            f"{random_run_num}_traj_{x}_smpl_{y}", random_match)
                    except:
                        continue

    else: # no samples in database so far, add each id once to get started 
        for x1 in range(num_of_traj):
            x2 = np.mod(x1+1, num_of_traj)
            for y in range(num_of_samples):
                print(x1,x2)
                db.insert_traj_pair(
                    f"{random_run_num}_traj_{x1}_smpl_{y}",
                    f"{random_run_num}_traj_{x2}_smpl_{y}"
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample clips from pretrained model')
    parser.add_argument("-r", "--run-id", required=True, help="ID for current run")
    parser.add_argument("-o", "--output-basedir", default="./output", help="Base dir for outputs")
    options = parser.parse_args()

    cfg = config.load_config(Path(options.output_basedir) / options.run_id / "run.yaml")

    # model_path = Path(cfg["out_models_dir"]) / "Q_0.pth"
    # assert model_path.exists()
    # print("Loading pretrained model from", model_path)
    # print("Running model in", cfg['env_task'])
    # print(f"Sampling {cfg['clip_sampler']['num_clips']} clips of length {cfg['clip_sampler']['clip_length']}")
    # print("Adding pairs of sampled clips to", cfg['out_annotation_db'])
    # with open(cfg['out_annotation_db'], 'w') as f:
    #     f.write("DUMMY DB")

    print(f"Initializing environment {cfg['env_task']}. This might take a while...")
    environment = gym.make(cfg['env_task'])
    print("Done initializing environment!")

    db = database.AnnotationBuffer()
    os.makedirs("trajectories", exist_ok=True)
    random_run_num = random.randint(100000, 999999) # this is for getting new file names for each run
    
    rng = np.random.default_rng(12345)
    print(len(list(range(cfg['num_of_traj']))))
    for traj_idx in tqdm(range(cfg['num_of_traj'])):
        trajectory = generate_trajectory(cfg['max_traj_length'],environment)
        save_trajectory(traj_idx, trajectory, random_run_num)

        for sample_idx in range(cfg['num_of_samples']):
            sample = generate_sample(trajectory, cfg['max_traj_length'], cfg['sample_length'] )
            save_sample(traj_idx, sample_idx,sample, random_run_num)
            fill_database(db, cfg['num_of_traj'], cfg['num_of_samples'], random_run_num, cfg['pair_per_sample'])
