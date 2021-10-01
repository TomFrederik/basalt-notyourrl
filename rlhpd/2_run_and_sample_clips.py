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


class DataBaseFiller:
    """
    Fills the database during one round, reinitialise it every round.

    It will generate num_traj number of trajectories, each of maximum length max_traj_length
    Each trajectory will be sampled num_samples times, of length sample_length
    Each sample will be paired with pair_per_sample number of randomly selected clips
    (except if database is empty, in that case every sample will be added once)  

    Set autolabel if you also want your samples to be paired with num_autolabel number of 
    demonstration samples.
    """
    def __init__(self, cfg, env, autolabel= True, autolabel_pair_per_sample=1):

        self.env = env
        self.rng = np.random.default_rng(cfg.sampler.rnd_seed)  # TODO JUN, is this okay here?
        
        self.autolabel = autolabel
        self.autolabel_num = autolabel_pair_per_sample
        
        self.num_traj = cfg.sampler.num_traj
        self.max_traj_length = cfg.sampler.max_traj_length
        self.num_samples = cfg.sampler.num_samples
        self.sample_length = cfg.sampler.sample_length
        self.pair_per_sample = cfg.sampler.pair_per_sample
        self.traj_dir = Path(cfg.sampler.traj_dir)

        self.db_path = Path(cfg.sampler.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = database.AnnotationBuffer(self.db_path)
        os.makedirs(self.traj_dir, exist_ok=True)
        self.run_id = time.strftime('%Y%m%d-%H%M%S')


    def _step_random_policy(self):
        """ Stepping on a random policy"""
        action = self.env.action_space.sample()
        observation, _, done, _ = self.env.step(action)
        return observation, done


    def _generate_trajectory(self):
        """ Generates a trajectory with a random policy at the moment""" #TODO pass a policy in, currently random
        done = False
        observation = self.env.reset()

        trajectory = np.zeros(shape=(self.max_traj_length, 64, 64, 3))
        step_idx = 0
        while not done and step_idx < self.max_traj_length:
            trajectory[step_idx] = observation["pov"]
            observation, done = self._step_random_policy()
            step_idx += 1
            self.env.render()
        return trajectory

    def _generate_sample(self, trajectory):
        """ 
        Generates a random sample of sample_length from a trajectory.
        Both trajectory and sample are numpy arrays
        """
        assert self.max_traj_length > self.sample_length
        starting_idx = self.rng.integers(low=0, high=self.max_traj_length-self.sample_length)
        sample = trajectory[starting_idx:starting_idx+self.sample_length,...]
        return sample

    @staticmethod
    def _write_to_file(path, np_array):
        np.save(path, np_array)

    def _save_all_traj_and_samples(self):
        """ Saves all the trajectories and samples as np array in a filename like 'runid_traj_x_smpl_y' """
        for traj_idx in tqdm(range(self.num_traj)):
            trajectory = self._generate_trajectory(self.max_traj_length, self.env)
            traj_path = self.traj_dir / f"{self.run_id}_traj_{traj_idx}_full"
            self._write_to_file(traj_path, trajectory)

            for sample_idx in range(self.num_samples):
                sample = self._generate_sample(trajectory, self.max_traj_length, self.sample_length)
                sample_path = self.traj_dir / f"{self.run_id}_traj_{traj_idx}_smpl_{sample_idx}"
                self._write_to_file(sample_path, sample)

    def _fill_database_from_files(self):
        """ Fills the database from the saved samples """
        #names of saved samples will be traj_x_smpl_y
        existing_ids =self.db.return_all_ids()
        if existing_ids:
            for x in range(self.num_traj):
                for y in range(self.num_samples):
                    for _ in range(self.pair_per_sample):
                        random_match = random.choice(existing_ids)[0]
                        try: # if we picked a pair that exists we just skip for now TODO
                            self.db.insert_traj_pair(
                                f"{self.run_id}_traj_{x}_smpl_{y}", random_match)
                        except:
                            continue

        else: # no samples in database so far, add each id once to get started 
            for x1 in range(self.num_traj):
                x2 = np.mod(x1+1, self.num_traj)
                for y in range(self.num_samples):
                    self.db.insert_traj_pair(
                        f"{self.run_id}_traj_{x1}_smpl_{y}",
                        f"{self.run_id}_traj_{x2}_smpl_{y}"
                        )

    def _get_demo_sample():
        """Returns a random sample from a random demo trajectory in a numpy array"""
    # TODO
    
    def _do_autolabels(self):
        """ Pairs each newly generated sample with a random sample from the demonstrations"""
        for x in range(self.num_of_traj):
            for y in range(self.num_of_samples):
                for _ in range(self.autolabel_pair_per_sample):
                    # get a random demo sample
                    demo_sample = self._get_demo_sample()
                    demo_path = self.traj_dir / f"demo_{self.run_id}_traj_{x}_smpl_{y}"
                    # save it to a file
                    self._write_to_file(demo_path,demo_sample)
                    try: # if we picked a pair that exists we just skip
                        # write it to database and rate better
                        self.db.insert_traj_pair(
                            f"{self.run_id}_traj_{x}_smpl_{y}", f"demo_{self.run_id}_traj_{x}_smpl_{y}")
                        self.db.rate_traj_pair(
                            f"{self.run_id}_traj_{x}_smpl_{y}", f"demo_{self.run_id}_traj_{x}_smpl_{y}", 2)
                    except:
                        continue


    def run(self):
        self._save_all_traj_and_samples()
        self._fill_database_from_files()
        if self.autolabel:
            self._do_autolabels()

# AUTOLABELING:
# prefer clips from demonstrations over initial trajectories
# pair up every initial trajectory with a random sample from a demonstration


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample clips from pretrained model')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    # Load config params
    cfg = utils.load_config(options.config_file)
    # Load env
    env_task = cfg.env_task
    print(f"Initializing environment {env_task}. This might take a while...")
    environment = gym.make(env_task)
    print("Done initializing environment!")

    db_filler = DataBaseFiller(cfg=cfg, env=environment, autolabel= True, autolabel_pair_per_sample=1)
    db_filler.run()

    

