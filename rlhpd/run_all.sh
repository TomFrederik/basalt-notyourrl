#!/bin/bash

CONFIG_FILE=config_BuildVillageHouse.yaml
python 1_pretrain.py -c $CONFIG_FILE
python 2_run.py -c $CONFIG_FILE
python 1_pretrain.py -c $CONFIG_FILE
python 1_pretrain.py -c $CONFIG_FILE
python 1_pretrain.py -c $CONFIG_FILE