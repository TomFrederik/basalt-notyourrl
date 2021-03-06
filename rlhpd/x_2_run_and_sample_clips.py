"""
Steps 3 & 4 in the algorithm:
Run pretrained model in environment to get trajectories
Sample clips from trajectories
Add clips (unannotated) into annotation db
"""

import argparse
import itertools
import pickle
import random
import time
from pathlib import Path

import gym
import minerl  # This is required to be able to import minerl environments
import numpy as np
import torch
from tqdm import tqdm

from .common import database, state_shaping, utils, DQfD_utils, action_shaping
from .common.DQfD_models import QNetwork


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
        self.max_pairs_per_autolabel_type = cfg.sampler.max_pairs_per_autolabel_type
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
        # Q_pre
        self.q_pre_path = cfg.pretrain_dqfd_args.model_path

        self.run_id = time.strftime('%Y%m%d-%H%M%S')

    def _generate_trajectory(self, policy=None, max_attempts=10):
        """
        Generates a trajectory with the given policy.
        Defaults to random policy if policy is None.
        Note: The trajectory may end prematurely depending on the policy actions.
        If the trajectory ends with fewer than sample_length frames, we try again
        up to max_attempts.
        """

        for _ in range(max_attempts):
            observation = self.env.reset()
            done = False
            trajectory = []
            for _ in range(self.traj_length):
                if policy == None: # Random policy
                    action = self.env.action_space.sample() # random policy
                else:
                    action = policy(observation)
                next_observation, reward, done, meta  = self.env.step(action)
                trajectory.append((observation, action, reward, next_observation, done, meta))

                observation = next_observation
                # self.env.render()
                if done:
                    break
            if len(trajectory) > self.sample_length:
                return trajectory
        raise Exception(f"Exceeded max attempts({max_attempts}) to generate trajectory of length {self.sample_length}")

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

    def _generate_demo_clips(self, save_dir):
        minerl_data = minerl.data.make(self.env_task, data_dir=self.demos_dir)
        traj_names = minerl_data.get_trajectory_names()

        save_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm(range(self.num_demo_clips)):
            random_traj = np.random.choice(traj_names)
            # minerl_data.load_data is very noisy, we suppress stdout here
            # with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            traj_frames = list(minerl_data.load_data(random_traj, include_metadata=True))
            # data_frames == list of (state, action, reward, next_state, done, meta)

            clip, start_idx = self._generate_sample(traj_frames, return_start_idx=True)
            # Mimics what the state_shaping.StateWrapper does, but for the demo states
            for frame in clip:
                state, action, reward, next_state, done, meta = frame
                frame[0] = state_shaping.preprocess_state(state, self.env_task)

            # Take note of the clip position within the trajectory; this is useful for
            # autolabelling early/late portions of the trajectory
            normalized_idx = start_idx / (len(traj_frames) - self.sample_length)
            percent = int(round(100 * normalized_idx))
            demo_path = save_dir / f"demo_{i:03d}_{random_traj}_{percent:03d}.pickle"
            print(demo_path)
            self._write_to_file(demo_path, clip)

    def _insert_all_pairs_into_db(self, good_paths, bad_paths, shuffle=True, max_pairs=None):
        # Generate all possible pairs
        all_pairs = list(itertools.product(good_paths, bad_paths))
        if shuffle:
            random.shuffle(all_pairs)
        if max_pairs is not None:
            # Limit the number of pairs
            all_pairs = all_pairs[:max_pairs]
        
        # Insert tuples in batches
        batch_of_traj_tuples = []
        for good_path, bad_path in tqdm(all_pairs):
            batch_of_traj_tuples.append([str(good_path), str(bad_path), 1])
            if len(batch_of_traj_tuples) == 1000:
                self.db.insert_many_traj_tuples(batch_of_traj_tuples)
                batch_of_traj_tuples = []
        # Insert any remainders
        self.db.insert_many_traj_tuples(batch_of_traj_tuples)
        return

    def _populate_db_autolabel_simple(self, good_dir: Path, bad_dir: Path, max_pairs=None):
        """
        Creates auto-labeled entries of preference pairs in the database,
        by pairing up random pairs of (bad_dir_sample, good_dir_sample).
        Label is such that good_dir samples are always preferred over bad_dir samples.
        """
        good_paths = sorted([x for x in good_dir.glob("*.pickle")])
        bad_paths = sorted([x for x in bad_dir.glob("*.pickle")])
        self._insert_all_pairs_into_db(good_paths, bad_paths, shuffle=True, max_pairs=max_pairs)
        return

    def _populate_db_autolabel_custom(self, clips_dir: Path, is_good, is_bad, max_pairs=None):
        """
        Creates auto-labeled entries of preference pairs in the database, by
        evaluating the given boolean functions `is_good` or `is_bad` on all clips
        inside clips_dir and populating the DB with every possible pairing of 
        (good, bad) samples.
        """
        all_paths = sorted([x for x in clips_dir.glob("*.pickle")])
        good_paths = []
        bad_paths = []
        # First sort paths into good, bad, or none
        for path in all_paths:
            if is_good(path):
                good_paths.append(path)
            elif is_bad(path):
                bad_paths.append(path)
        # Populate DB with every possible pairing of (good, bad)
        self._insert_all_pairs_into_db(good_paths, bad_paths, shuffle=True, max_pairs=max_pairs)
        return 

    def _populate_db_autolabel_early_late(self, clips_dir, max_pairs=None):
        """
        Clips that start earlier than 30% of the full trajectory are labelled as "worse"
        than clips that start later than 70% of the full trajectory. This implementation
        allows clips to be compared across different trajectories.
        """

        def is_early_clip(clip_path):
            position_percent = int(Path(clip_path).stem.split("_")[-1])
            if position_percent < 30:
                return True
            return False

        def is_late_clip(clip_path):
            position_percent = int(Path(clip_path).stem.split("_")[-1])
            if position_percent >= 70:
                return True
            return False

        return self._populate_db_autolabel_custom(
            clips_dir,
            is_good = is_late_clip,
            is_bad = is_early_clip,
            max_pairs = max_pairs,
        )

    def load_policy(self, model_path):
        # load model
        vec_sample = self.env.observation_space.sample()['vec']
        vec_dim = vec_sample.shape[0]
        print(f'vec_dim = {vec_dim}')

        num_actions = (len(action_shaping.INVENTORY[self.env_task]) + 1) * 360
        print(f'num_actions = {num_actions}')
        q_net = QNetwork(num_actions, vec_dim)
        q_net.load_state_dict(torch.load(model_path))
        q_net.eval()
        return q_net 

    def run(self):
        random_clips_dir = self.clips_dir / "random"
        demo_clips_dir = self.clips_dir / "demos"
        q_pre_clips_dir = self.clips_dir / "q_pre"

        print("Generating demo clips")
        self._generate_demo_clips(demo_clips_dir)

        print("Generating random clips")
        self._generate_policy_clips(random_clips_dir)

        print("Generating Q_pre policy clips")
        self.q_pre_model = self.load_policy(self.q_pre_path)
        def q_pre_policy(obs, epsilon=0):
            # extract pov and inv from obs and convert to torch tensors
            pov = torch.from_numpy(obs['pov'][None])
            vec = torch.from_numpy(obs['vec'][None])
            # compute q_values
            q_values = self.q_pre_model.forward(pov, vec)
            # select action
            if random.random() < epsilon:
                action = random.randint(0, self.q_pre_model.num_actions)
            else:
                action = torch.argmax(q_values).squeeze().item()
            return action
        self._generate_policy_clips(q_pre_clips_dir, policy_model=q_pre_policy)

        print("Running autolabeling...")
        # Random < any demos
        self._populate_db_autolabel_simple(good_dir = demo_clips_dir,
                                           bad_dir = random_clips_dir,
                                           max_pairs=self.max_pairs_per_autolabel_type)
        # Early < Late
        self._populate_db_autolabel_early_late(demo_clips_dir, max_pairs=self.max_pairs_per_autolabel_type)
        # Q_pre < any demos
        self._populate_db_autolabel_simple(good_dir = demo_clips_dir,
                                           bad_dir = q_pre_clips_dir,
                                           max_pairs=self.max_pairs_per_autolabel_type)

    
def main(cfg):

    # Load env
    env_name = cfg.env_task
    print(f"Initializing environment {env_name}. This might take a while...")
    env = gym.make(env_name)
    env = state_shaping.StateWrapper(env, env_name)
    env = DQfD_utils.RewardActionWrapper(env, env_name, DQfD_utils.DummyRewardModel())
    print("Done initializing environment!")

    db_filler = DataBaseFiller(cfg=cfg, env=env)
    db_filler.run()
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample clips from pretrained model')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()
    
    # Load config params
    cfg = utils.load_config(options.config_file)
    
    main(cfg)