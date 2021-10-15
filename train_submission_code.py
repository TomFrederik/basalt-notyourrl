import logging
import os
import shutil
from pathlib import Path

import coloredlogs
import gym
import minerl
import numpy as np

import aicrowd_helper
# from basalt_baselines.bc import bc_baseline
from utility.parser import Parser

coloredlogs.install(logging.DEBUG)

import json

from rlhpd import (DQfD_pretraining, DQfD_training, x_2_run_and_sample_clips,
                   x_3_train_reward)
from rlhpd.common import utils

# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')
# You need to ensure that your submission is trained within allowed training time.
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4 * 24 * 60))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))

BASALT_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLBasaltFindCave-v0')


def main(task_name = None):
    if task_name is None:
        with open("aicrowd.json",) as f:
            ai_crowd_json = json.load(f)
            task_name = ai_crowd_json["tags"]
    
    if task_name == "BuildVillageHouse":
        cfg = utils.load_config("rlhpd/config_BuildVillageHouse.yaml")
    elif task_name == "MakeWaterfall":
        cfg = utils.load_config("rlhpd/config_MakeWaterfall.yaml")
    elif task_name == "FindCave":
        cfg = utils.load_config("rlhpd/config_FindCave.yaml")
    elif task_name == "CreateVillageAnimalPen":
        cfg = utils.load_config("rlhpd/config_VillageAnimalPen.yaml")
    elif task_name == "MakeWaterfall_small":
        cfg = utils.load_config("rlhpd/config_MakeWaterfall_small.yaml")
    else: 
        Exception("tag is wrong")
    model_savepath = (Path("train") / cfg.env_task).with_suffix(".pt")
    
    # Pretrain
    print("############################## Running DQfD pretraining!")
    args = {**vars(cfg.pretrain_dqfd_args), **vars(cfg.dqfd_args)} # join common dqfd args with those that are specific for pretraining
    DQfD_pretraining.main(**args)
    shutil.copyfile(cfg.pretrain_dqfd_args.model_path, model_savepath)
    print(f"Copied {cfg.pretrain_dqfd_args.model_path} to {model_savepath}")

    # # Run and sample
    # print("############################## Sampling clips")
    # x_2_run_and_sample_clips.main(cfg)

    # # Train reward
    # print("############################## Training reward!")
    # x_3_train_reward.main(cfg)

    # # Train DQfD
    # print("############################## Running DQfD training!")
    # args = {**vars(cfg.train_dqfd_args), **vars(cfg.dqfd_args)} # join common dqfd args with those that are specific for training
    # DQfD_training.main(**args)
    # shutil.copyfile(cfg.train_dqfd_args.new_model_path, model_savepath)
    # print(f"Copied {cfg.train_dqfd_args.new_model_path} to {model_savepath}")

    # # Advanced: Keep looping
    # for i in range(cfg.num_big_loops):
        
    #     # Prep for new run
    #     def add_number_to_path_stem(filepath: str, num: int):
    #         """
    #         Converts a filepath like
    #         path/to/my/file.suffix into 
    #         path/to/my/file_001.suffix
    #         """
    #         return str(Path(filepath).parent / Path(filepath).stem) + f"_{num:03d}.pt"
    #     # Add counter to each of the model paths
    #     cfg.pretrain_dqfd_args.model_path = add_number_to_path_stem(cfg.pretrain_dqfd_args.model_path, i)
    #     cfg.train_dqfd_args.model_path = add_number_to_path_stem(cfg.train_dqfd_args.model_path, i)
    #     cfg.train_dqfd_args.new_model_path = add_number_to_path_stem(cfg.train_dqfd_args.new_model_path, i)
    #     shutil.rmtree(cfg.sampler.db_path)
    #     shutil.rmtree(cfg.sampler.clips_dir)

    #     # Run and sample
    #     print(f"############################## Sampling clips {i+1}")
    #     x_2_run_and_sample_clips.main(cfg)

    #     # Train reward
    #     print(f"############################## Training reward! {i+1}")
    #     x_3_train_reward.main(cfg)

    #     # Train DQfD
    #     print(f"############################## Running DQfD training! {i+1}")
    #     args = {**vars(cfg.train_dqfd_args), **vars(cfg.dqfd_args)} # join common dqfd args with those that are specific for training
    #     DQfD_training.main(**args)
    #     shutil.copyfile(cfg.train_dqfd_args.new_model_path, model_savepath)
    #     print(f"Copied {cfg.train_dqfd_args.new_model_path} to {model_savepath}")

    print(f"Training for {task_name} ran successfully!")

if __name__ == "__main__":
    main()

# # Optional: You can view best effort status of your instances with the help of parser.py
# # This will give you current state like number of steps completed, instances launched and so on.
# # Make your you keep a tap on the numbers to avoid breaching any limits.
# parser = Parser(
#     'performance/',
#     maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
#     raise_on_error=False,
#     no_entry_poll_timeout=600,
#     submission_timeout=MINERL_TRAINING_TIMEOUT * 60,
#     initial_poll_timeout=600
# )


# def basic_train():
#     """
#     This function will be called for training phase.
#     This should produce and save same files you upload during your submission.
#     """
#     # How to sample minerl data is document here:
#     # http://minerl.io/docs/tutorials/data_sampling.html
#     data = minerl.data.make('MineRLBasaltFindCave-v0', data_dir=MINERL_DATA_ROOT)

#     # Sample code for illustration, add your training code below
#     env = gym.make('MineRLBasaltFindCave-v0')

#     # For an example, lets just run one episode of MineRL for training
#     obs = env.reset()
#     done = False
#     while not done:
#         obs, reward, done, info = env.step(env.action_space.sample())
#         # Do your training here

#         # To get better view in your training phase, it is suggested
#         # to register progress continuously, example when 54% completed
#         # aicrowd_helper.register_progress(0.54)

#         # To fetch latest information from instance manager, you can run below when you want to know the state
#         #>> parser.update_information()
#         #>> print(parser.payload)

#     # Save trained model to train/ directory
#     # For a demonstration, we save some dummy data.
#     np.save("./train/parameters.npy", np.random.random((10,)))

#     # Training 100% Completed
#     aicrowd_helper.register_progress(1)
#     env.close()


# def main():
#     # Documentation for BC Baseline can be found in train_bc.py
#     # TODO make this configurable once we have multiple baselines
#     TRAINING_EXPERIMENT = bc_baseline
#     TRAINING_EXPERIMENT.run(config_updates={'data_root': MINERL_DATA_ROOT,
#                                             'task_name': BASALT_GYM_ENV,
#                                             'train_batches': 10,
#                                             'save_dir': "train"})
