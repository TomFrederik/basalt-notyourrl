#!/bin/bash

ENV_NAME=MineRLBasaltFindCave-v0
NUM_EPISODES=5
MODEL=Q_0
python rlhpd/DQfD_eval_pretrained.py --env_name $ENV_NAME  --model_path output/$ENV_NAME/$MODEL.pth --video_dir output/$ENV_NAME/videos/${MODEL} --num_episodes $NUM_EPISODES


ENV_NAME=MineRLBasaltFindCave-v0
NUM_EPISODES=5
MODEL=Q_1
python rlhpd/DQfD_eval_pretrained.py --env_name $ENV_NAME  --model_path output/$ENV_NAME/$MODEL.pth --video_dir output/$ENV_NAME/videos/${MODEL} --num_episodes $NUM_EPISODES


ENV_NAME=MineRLBasaltMakeWaterfall-v0
NUM_EPISODES=5
MODEL=Q_0
python rlhpd/DQfD_eval_pretrained.py --env_name $ENV_NAME  --model_path output/$ENV_NAME/$MODEL.pth --video_dir output/$ENV_NAME/videos/${MODEL} --num_episodes $NUM_EPISODES


ENV_NAME=MineRLBasaltMakeWaterfall-v0
NUM_EPISODES=5
MODEL=Q_1
python rlhpd/DQfD_eval_pretrained.py --env_name $ENV_NAME  --model_path output/$ENV_NAME/$MODEL.pth --video_dir output/$ENV_NAME/videos/${MODEL} --num_episodes $NUM_EPISODES
