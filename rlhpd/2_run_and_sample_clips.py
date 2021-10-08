"""
Steps 3 & 4 in the algorithm:
Run pretrained model in environment to get trajectories
Sample clips from trajectories
Add clips (unannotated) into annotation db
"""

import argparse
import os
import pickle
import random
import time
from pathlib import Path

import gym
import minerl  # This is required to be able to import minerl environments
import numpy as np
from tqdm import tqdm

from common import database, utils, DQfD_utils


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
    def __init__(self, cfg, env):

        self.env = env
        self.env_task = cfg.env_task
        self.rng = np.random.default_rng(cfg.sampler.rnd_seed)  
        
        self.autolabel_with_demos = cfg.sampler.autolabel_with_demos
        self.autolabel_demo_num = cfg.sampler.autolabel_demo_per_sample
        self.autolabel_early_late_demos = cfg.sampler.autolabel_early_late_demos
        # self.policy_model = load_policy(cfg.pretrain_dqfd_args.model_path)
        
        self.num_traj = cfg.sampler.num_traj
        self.max_traj_length = cfg.sampler.max_traj_length
        self.num_samples = cfg.sampler.num_samples
        self.sample_length = cfg.sampler.sample_length
        #self.pair_per_sample = cfg.sampler.pair_per_sample
        self.traj_dir = Path(cfg.sampler.traj_dir)
        self.demos_dir = cfg.demos_dir
        self.db_path = Path(cfg.sampler.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = database.AnnotationBuffer(self.db_path)
        os.makedirs(self.traj_dir, exist_ok=True)
        self.run_id = time.strftime('%Y%m%d-%H%M%S')


    def _random_policy(self, _):
        """ Stepping on a random policy"""
        return self.env.action_space.sample()

    # def _learned_policy(self, obs):
    #     """ Stepping policy"""
    #     action = self.policy_model(obs)
    #     return action

    def _generate_trajectory(self):
        """ Generates a trajectory with a random policy at the moment""" #TODO pass a policy in, currently random
        done = False
        observation = self.env.reset()

        trajectory = []
        for i in range(self.max_traj_length):
            # trajectory[step_idx] = observation["pov"]
            # observation, _, done, _ = self._step_random_policy()
            action = self._random_policy(observation)
            next_observation, reward, done, meta  = self.env.step(action)
            trajectory.append((observation, action, reward, next_observation, done, meta))

            observation = next_observation

            self.env.render()
            # if done:
            #     break
        print("Generated trajectory of length", len(trajectory))
        return trajectory

    def _generate_sample(self, trajectory):
        """ 
        Generates a random sample of sample_length from a trajectory.
        Both trajectory and sample are numpy arrays
        """
        assert self.max_traj_length > self.sample_length
        starting_idx = self.rng.integers(low=0, high=self.max_traj_length-self.sample_length)
        sample = trajectory[starting_idx : starting_idx+self.sample_length]
        return sample

    @staticmethod
    def _write_to_file(path, trajectory):
        print(f"Saving trajectory {path} of size", len(trajectory))
        with open(path, 'wb') as f:
            pickle.dump(trajectory, f)

    def _save_all_traj_and_samples(self):
        """ Saves all the trajectories and samples as np array in a filename like 'runid_traj_x_smpl_y' """
        for traj_idx in tqdm(range(self.num_traj)):
            trajectory = self._generate_trajectory()
            traj_path = self.traj_dir / f"{self.run_id}_traj_{traj_idx}_full.pickle"
            self._write_to_file(traj_path, trajectory)

            for sample_idx in range(self.num_samples):
                sample = self._generate_sample(trajectory)
                sample_path = self.traj_dir / f"{self.run_id}_traj_{traj_idx}_smpl_{sample_idx}.pickle"
                self._write_to_file(sample_path, sample)

    # def _fill_database_from_files(self):
    #     """ Fills the database from the saved samples """
    #     #names of saved samples will be traj_x_smpl_y
    #     existing_ids =self.db.return_all_ids()
    #     if existing_ids:
    #         for x in range(self.num_traj):
    #             for y in range(self.num_samples):
    #                 for _ in range(self.pair_per_sample):
    #                     random_match = random.choice(existing_ids)[0]
    #                     try: # if we picked a pair that exists we just skip for now TODO
    #                         self.db.insert_traj_pair(
    #                             f"{self.run_id}_traj_{x}_smpl_{y}", random_match)
    #                     except:
    #                         continue

        # else: # no samples in database so far, add each id once to get started 
        #     for x1 in range(self.num_traj):
        #         x2 = np.mod(x1+1, self.num_traj)
        #         for y in range(self.num_samples):
        #             self.db.insert_traj_pair(
        #                 f"{self.run_id}_traj_{x1}_smpl_{y}",
        #                 f"{self.run_id}_traj_{x2}_smpl_{y}"
        #    
        # 
        #              )

    def _get_demo_sample(self):
        """
        Returns a random sample from a random demo trajectory in a numpy array, and the end of trajectory
        sample from the same trajectory
        """
        minerl_data = minerl.data.make(self.env_task, data_dir=self.demos_dir)
        traj_names = minerl_data.get_trajectory_names()
        random_traj = np.random.choice(traj_names)
        data_frames = list(minerl_data.load_data(random_traj, include_metadata=True))
        # data_frames == list of (state, action, reward, next_state, done, meta)

        start_idx = self.rng.integers(low=0, high=len(data_frames)-self.sample_length-1)
        random_clip = data_frames[start_idx: start_idx + self.sample_length]
        end_clip = data_frames[len(data_frames)-self.sample_length: len(data_frames)]
        # Insert flattened vector representation of dictionary states
        # Mimics what the DQfD_utils.StateWrapper does, but for the demo actions
        for frame in random_clip:
            state, action, reward, next_state, done, meta = frame
            frame[0]['vec'] = DQfD_utils.preprocess_non_pov_obs(state)
        for frame in end_clip:
            state, action, reward, next_state, done, meta = frame
            frame[0]['vec'] = DQfD_utils.preprocess_non_pov_obs(state)
        return random_clip, end_clip
    
    def _do_autolabels(self):
        """ Pairs each newly generated sample with a random sample from the demonstrations"""
        for x in range(self.num_traj):
            for y in range(self.num_samples):
                for _ in range(self.autolabel_demo_num):
                    # get a random demo sample
                    demo_sample, end_clip = self._get_demo_sample()
                    demo_path = self.traj_dir / f"demo_{self.run_id}_traj_{x}_smpl_{y}.pickle"
                    # save it to a file
                    self._write_to_file(demo_path,demo_sample)
                    if self.autolabel_early_late_demos:
                        end_path = self.traj_dir / f"demo_{self.run_id}_traj_{x}_smpl_{y}_end.pickle"
                        self._write_to_file(end_path,end_clip)
                    try: # if we picked a pair that exists we just skip
                        # write it to database and rate better
                        self.db.insert_traj_pair(
                            f"{self.run_id}_traj_{x}_smpl_{y}", f"demo_{self.run_id}_traj_{x}_smpl_{y}")
                        self.db.rate_traj_pair(
                            f"{self.run_id}_traj_{x}_smpl_{y}", f"demo_{self.run_id}_traj_{x}_smpl_{y}", 2)
                        if self.autolabel_early_late_demos:
                            self.db.insert_traj_pair(
                            f"demo_{self.run_id}_traj_{x}_smpl_{y}", f"demo_{self.run_id}_traj_{x}_smpl_{y}_end")
                            self.db.rate_traj_pair(
                            f"demo_{self.run_id}_traj_{x}_smpl_{y}", f"demo_{self.run_id}_traj_{x}_smpl_{y}_end", 2)
                    except:
                        continue

    def run(self):
        print("Generating clips from policy...")
        self._save_all_traj_and_samples()
        #print("Adding clips to database...")
        #self._fill_database_from_files()
        if self.autolabel_with_demos:
            print("Adding autolabelled clips from demonstrations...")
            self._do_autolabels()


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
    env = gym.make(env_task)
    env = DQfD_utils.StateWrapper(env)
    print("Done initializing environment!")

    db_filler = DataBaseFiller(cfg=cfg, env=env)
    db_filler.run()

    

