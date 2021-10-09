"""
Steps 3 & 4 in the algorithm:
Run pretrained model in environment to get trajectories
Sample clips from trajectories
Add clips (unannotated) into annotation db
"""

import argparse
import contextlib
import os
import pickle
import random
import time
from pathlib import Path

import gym
import minerl  # This is required to be able to import minerl environments
import numpy as np
from tqdm import tqdm

from common import database, state_shaping, utils


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
        # Load params from config
        self.env = env
        self.env_task = cfg.env_task
        utils.set_seeds(cfg.sampler.rnd_seed)
        # Database to record preference labels
        self.db_path = Path(cfg.sampler.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = database.AnnotationBuffer(self.db_path)
        # All clips
        self.sample_length = cfg.sampler.sample_length
        self.traj_dir = Path(cfg.sampler.traj_dir)
        self.traj_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir = Path(cfg.sampler.clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        # Random
        self.num_traj = cfg.sampler.num_traj
        self.traj_length = cfg.sampler.traj_length
        self.num_random_clips_per_traj = cfg.sampler.num_random_clips_per_traj
        # Demos
        self.num_demo_clips = cfg.sampler.num_demo_clips
        self.demos_dir = Path(cfg.demos_dir)
        self.demos_dir.mkdir(parents=True, exist_ok=True)

        # self.autolabel_with_demos = cfg.sampler.autolabel_with_demos
        # self.autolabel_demo_num = cfg.sampler.autolabel_demo_per_sample
        # self.autolabel_early_late_demos = cfg.sampler.autolabel_early_late_demos
        # self.pair_per_sample = cfg.sampler.pair_per_sample

        self.run_id = time.strftime('%Y%m%d-%H%M%S')

    def _generate_trajectory(self, policy_model=None):
        """
        Generates a trajectory with the given policy_model.
        Defaults to random policy if policy_model is None.
        """
        done = False
        observation = self.env.reset()

        trajectory = []
        for i in range(self.traj_length):
            if policy_model == None: # Random policy
                action = self.env.action_space.sample() # random policy
            else:
                action = self.policy_model(observation)
            next_observation, reward, done, meta  = self.env.step(action)
            trajectory.append((observation, action, reward, next_observation, done, meta))

            observation = next_observation
            self.env.render()
            # if done:
            #     break
        return trajectory

    def _generate_sample(self, trajectory, return_start_idx=False):
        """ 
        Generates a random sample of sample_length from a trajectory.
        Both trajectory and sample are numpy arrays
        """
        assert len(trajectory) > self.sample_length
        starting_idx = np.random.randint(
            low = 0, 
            high = len(trajectory) - self.sample_length
        )
        sample = trajectory[starting_idx : starting_idx + self.sample_length]
        assert len(sample) == self.sample_length
        if return_start_idx:
            return sample, starting_idx
        return sample

    @staticmethod
    def _write_to_file(path, trajectory):
        with open(path, 'wb') as f:
            pickle.dump(trajectory, f)

    def _generate_policy_clips(self, save_dir, policy_model=None):
        """
        Generates random policy trajectories, and samples clips from them
        Saves all the trajectories and clips as np array in a filename like 'runid_traj_x_smpl_y'
        """
        traj_dir = save_dir / "source_traj"
        traj_dir.mkdir(parents=True, exist_ok=True)
        for traj_idx in tqdm(range(self.num_traj)):
            trajectory = self._generate_trajectory(policy_model)
            traj_path = traj_dir / f"{self.run_id}_traj_{traj_idx}_full.pickle"
            self._write_to_file(traj_path, trajectory)

            for sample_idx in range(self.num_random_clips_per_traj):
                sample = self._generate_sample(trajectory)
                sample_path = save_dir / f"{self.run_id}_traj_{traj_idx}_smpl_{sample_idx}.pickle"
                self._write_to_file(sample_path, sample)

    def _generate_demo_clips(self):
        minerl_data = minerl.data.make(self.env_task, data_dir=self.demos_dir)
        traj_names = minerl_data.get_trajectory_names()

        clips_dir = self.clips_dir / "demos"
        clips_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(self.num_demo_clips)):
            random_traj = np.random.choice(traj_names)
            # minerl_data.load_data is very noisy, we suppress stdout here
            # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            traj_frames = list(minerl_data.load_data(random_traj, include_metadata=True))
            # data_frames == list of (state, action, reward, next_state, done, meta)

            clip, start_idx = self._generate_sample(traj_frames, return_start_idx=True)
            # Insert flattened vector representation of dictionary states
            # Mimics what the state_shaping.StateWrapper does, but for the demo actions
            for frame in clip:
                state, action, reward, next_state, done, meta = frame
                frame[0]['vec'] = state_shaping.preprocess_non_pov_obs(state)

            # Take note of the clip position within the trajectory; this is useful for
            # autolabelling early/late portions of the trajectory
            normalized_idx = start_idx / (len(traj_frames) - self.sample_length)
            percentile = int(10 * np.floor(100 * normalized_idx / 10))
            demo_path = clips_dir / f"demo_{i:03d}_{random_traj}_{percentile:03d}.pickle"
            self._write_to_file(demo_path, clip)

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

    def _populate_db_autolabel_simple(self, good_dir: Path, bad_dir: Path):
        """
        Creates auto-labeled entries of preference pairs in the database,
        by pairing up random pairs of (bad_dir_sample, good_dir_sample).
        Label is such that good_dir samples are always preferred over bad_dir samples.
        """
        good_paths = sorted([x for x in good_dir.glob("*.pickle")])
        bad_paths = sorted([x for x in bad_dir.glob("*.pickle")])
        for good_path in tqdm(good_paths):
            batch_of_traj_tuples = []
            for bad_path in bad_paths:
                batch_of_traj_tuples.append((str(good_path), str(bad_path), 1))
            self.db.insert_many_traj_tuples(batch_of_traj_tuples)
        return
        
    def run(self):
        # print("Generating clips from policy...")
        # self._save_all_traj_and_samples()
        # #print("Adding clips to database...")
        # #self._fill_database_from_files()
        # if self.autolabel_with_demos:
        #     print("Adding autolabelled clips from demonstrations...")
        #     self._do_autolabels()
        
        random_clips_dir = self.clips_dir / "random"
        demo_clips_dir = self.clips_dir / "demos"

        print("Generating demo clips")
        self._generate_policy_clips(random_clips_dir)
        print("Generating random clips")
        self._generate_demo_clips(demo_clips_dir)
        # print("Generating trained policy clips")
        # self.policy_model = load_policy(cfg.pretrain_dqfd_args.model_path)
        # self._generate_policy_clips(policy_name="Q_pre", policy_model=self.policy_model)

        print("Running autolabeling...")

        # Random < any demos
        self._populate_db_autolabel_simple(
            good_dir = demo_clips_dir,
            bad_dir = random_clips_dir,
        )

        # def is_early_clip(clipname):
        #     clipname.
        # populate_db_autolabel_with_lambdas(
        #     demo_clips_dir,
        #     is_worse = (lambda filename : filename.starts_with()), # 
        #     is_better,
        # )

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
    env = state_shaping.StateWrapper(env)
    print("Done initializing environment!")

    db_filler = DataBaseFiller(cfg=cfg, env=env)
    db_filler.run()

    

