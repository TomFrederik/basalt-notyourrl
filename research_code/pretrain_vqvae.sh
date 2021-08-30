# set parameters
NUM_EMBED = 512
EMBED_DIM = 32
N_HID = 64
BATCH_SIZE = 500
LOG_DIR = "/home/lieberummaas/datadisk/basalt-notyourrl/run_logs"
DATA_DIR = "/home/lieberummaas/datadisk/basalt-notyourrl/data"
ENV_NAME = "MineRLNavigate-v0"

# run training script
python3 -u vqvae.py --num_embeddings NUM_EMBED --embedding_dim EMBED_DIM --n_hid N_HID --batch_size BATCH_SIZE --log_dir LOG_DIR --data_dir DATA_DIR --env_name ENV_NAME
