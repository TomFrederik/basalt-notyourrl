#!/bin/bash

NUM_EPISODES=5

ENV_NAME=MineRLBasaltFindCave-v0
xvfb-run -a python rlhpd/DQfD_eval_pretrained.py --env_name $ENV_NAME  --model_path train/$ENV_NAME.pt --video_dir output/videos/$ENV_NAME --num_episodes $NUM_EPISODES

ENV_NAME=MineRLBasaltMakeWaterfall-v0
xvfb-run -a python rlhpd/DQfD_eval_pretrained.py --env_name $ENV_NAME  --model_path train/$ENV_NAME.pt --video_dir output/videos/$ENV_NAME --num_episodes $NUM_EPISODES

ENV_NAME=MineRLBasaltBuildVillageHouse-v0
xvfb-run -a python rlhpd/DQfD_eval_pretrained.py --env_name $ENV_NAME  --model_path train/$ENV_NAME.pt --video_dir output/videos/$ENV_NAME --num_episodes $NUM_EPISODES

ENV_NAME=MineRLBasaltCreateVillageAnimalPen-v0
xvfb-run -a python rlhpd/DQfD_eval_pretrained.py --env_name $ENV_NAME  --model_path train/$ENV_NAME.pt --video_dir output/videos/$ENV_NAME --num_episodes $NUM_EPISODES