```bash
python 1_pretrain.py
RLHPD_RUN_ID=run-20210914-152431.802 # Replace with your own run id
python 2_run_and_sample_clips.py -r $RLHPD_RUN_ID
python 3_train_reward.py -r $RLHPD_RUN_ID
python 4_train_dqfd.py -r $RLHPD_RUN_ID
```