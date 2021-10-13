#!/bin/bash

CONFIG_FILE=config_MakeWaterfall.yaml
python 1_pretrain.py -c $CONFIG_FILE
python 2_run_and_sample_clips.py -c $CONFIG_FILE
python 3_train_reward.py -c $CONFIG_FILE
streamlit run rate_clips.py -- -c $CONFIG_FILE