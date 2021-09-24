import argparse
import pickle
import random
from pathlib import Path

import einops
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import dataset
import model
import utils

def get_pred_accuracy(prefer_probs, true_judgements):
    pred_judgements = utils.probs_to_judgements(prefer_probs.detach().numpy())
    assert pred_judgements.shape == true_judgements.shape
    eq = np.all(true_judgements.numpy() == pred_judgements, axis=1)
    acc = eq.sum() / len(eq)
    return acc

def evaluate_model_accuracy(reward_model, dataloader, max_batches=None):
    total_answers = 0
    correct_answers = 0
    tqdm_total = len(dataloader) if max_batches is None else max_batches
    for batch_idx, batch in tqdm(enumerate(dataloader), total=tqdm_total):
        if max_batches is not None and batch_idx >= max_batches:
            break
        frames_a, frames_b, true_judgements = \
            batch['frames_a'], batch['frames_b'], batch['judgement']
        # Use model to predict probabilities of each judgement
        probs = utils.predict_pref_probs(reward_model, frames_a, frames_b)
        # Round probabilities to the closest judgement
        pred_judgements = utils.probs_to_judgements(probs.detach().numpy())
        assert pred_judgements.shape == true_judgements.shape
        # Compare prediction to true judgements
        eq = np.all(true_judgements.numpy() == pred_judgements, axis=1)
        correct_answers += eq.sum()
        total_answers += len(eq)
    mean_acc = correct_answers / total_answers
    return mean_acc

def set_seeds(num):
    torch.manual_seed(num)
    random.seed(num)
    np.random.seed(num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model')
    parser.add_argument("-c", "--clips-dir", default="./output",
                        help="Clips directory. Default: %(default)s")
    parser.add_argument("-s", "--save-root-dir", default="./models",
                        help="Root directory to save models. Default: %(default)s")
    options = parser.parse_args()

    wandb.init(project='DRLfHP-cartpole', entity='junshern')
    # wandb.init(project='DRLfHP-cartpole', entity='junshern', mode="disabled")

    save_dir = Path(options.save_root_dir) / wandb.run.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # The model is trained on batches of 16 segment pairs (see below), 
    # optimized with Adam (Kingma and Ba, 2014) 
    # with learning rate 0.0003, β1 = 0.9, β2 = 0.999, and eps = 10^−8 .
    cfg = wandb.config
    cfg.adam_lr = 0.0003
    cfg.adam_betas = (0.9, 0.999)
    cfg.adam_eps = 1e-8
    cfg.batch_size = 16
    cfg.max_num_pairs = None
    cfg.rand_seed = 0
    cfg.val_split = 0.1
    cfg.val_every = 50
    cfg.save_every = 100

    set_seeds(cfg.rand_seed)

    reward_model = model.RewardModel()
    wandb.watch(reward_model)
    optimizer = torch.optim.Adam(
        reward_model.parameters(),
        lr=cfg.adam_lr,
        betas=cfg.adam_betas,
        eps=cfg.adam_eps)

    traj_paths = sorted([x for x in Path(options.clips_dir).glob("*.pickle")])

    # Check the length of a single clip
    with open(traj_paths[0], 'rb') as f:
        clip = pickle.load(f)
        cfg.clip_length = len(clip)

    # resize_imgs = transforms.Compose([transforms.Resize((64, 64))])
    full_dataset = dataset.TrajectoryPreferencesDataset(options.clips_dir)
    val_size = int(cfg.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0)

    for batch_idx, data_batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        frames_a, frames_b, judgements = \
            data_batch['frames_a'], data_batch['frames_b'], data_batch['judgement']

        # Run prediction pipeline
        prefer_probs = utils.predict_pref_probs(reward_model, frames_a, frames_b)

        # Calculate loss:
        # This is ALMOST CrossEntropy but slightly different because we want to support
        # tie condition (0.5, 0.5) in judgement which is not possible in default XEnt
        # loss = - (judgement[0] * torch.log(prefer_1) + judgement[1] * torch.log(prefer_2))
        assert judgements.shape == prefer_probs.shape
        loss = - torch.sum(judgements * torch.log(prefer_probs))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation & logging
        with torch.no_grad():
            reward_model.eval()
            wandb.log({"loss": loss.item()})
            train_acc = get_pred_accuracy(prefer_probs, judgements)
            wandb.log({"train_acc": train_acc})
            if batch_idx % cfg.val_every == 0:
                # Validation accuracy
                val_acc = evaluate_model_accuracy(reward_model, val_dataloader)
                wandb.log({"val_acc": val_acc})
            reward_model.train()
        
        # Save model
        if batch_idx % cfg.save_every == 0:
            save_path = save_dir / f"{batch_idx:05d}.pt"
            torch.save(reward_model.state_dict(), save_path)
            print("Saved model to", save_path)

        # TODO: A fraction of 1/e of the data is held out to be used as a validation set. 
        # We use L2- regularization of network weights with the adaptive scheme described in 
        # Christiano et al. (2017): the L2-regularization weight increases if the average 
        # validation loss is more than 50% higher than the average training loss, and decreases 
        # if it is less than 10% higher (initial weight 0.0001, multiplicative rate of change 
        # 0.001 per learning step).

        # TODO: An extra loss proportional to the square of the predicted rewards is added 
        # to impose a zero-mean Gaussian prior on the reward distribution.

        # TODO: Gaussian noise of amplitude 0.1 (the grayscale range is 0 to 1) is added to the inputs.
        # TODO: Grayscale images?

        # TODO: We assume there is a 10% chance that the annotator responds uniformly at random, 
        # so that the model will not overfit to possibly erroneous preferences. We account for 
        # this error rate by using Pˆ e = 0.9Pˆ + 0.05 instead of Pˆ for the cross-entropy computation.

        # TODO: since the reward model is trained merely on comparisons, its absolute scale is arbitrary. 
        # Therefore we normalize its output so that it has 0 mean and standard deviation 0.05 
        # across the annotation buffer
        # JS: This is only when passing as output to the RL algorithm?
        # See https://github.com/mrahtz/learning-from-human-preferences/blob/master/reward_predictor.py#L167-L169

    # Save final model
    save_path = save_dir / f"{batch_idx:05d}.pt"
    torch.save(reward_model.state_dict(), save_path)
    print("Saved model to", save_path)
