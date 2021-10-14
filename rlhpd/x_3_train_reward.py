"""
Step 7 in the algorithm:
Train reward model with preferences from the annotation buffer
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import preference_helpers as pref
from .common import utils
from .common.dataset import TrajectoryPreferencesDataset
from .common.reward_model import RewardModel


def get_pred_accuracy(prefer_probs, true_judgements):
    pred_judgements = pref.probs_to_judgements(prefer_probs.detach().numpy())
    assert pred_judgements.shape == true_judgements.shape
    eq = np.all(true_judgements.numpy() == pred_judgements, axis=1)
    acc = eq.sum() / len(eq)
    return acc

def evaluate_model_accuracy(reward_model, dataloader, max_batches=None, ret_rewards=False):
    total_answers = 0
    correct_answers = 0
    all_pred_rewards = [] # Save pred rewards for analysis
    tqdm_total = len(dataloader) if max_batches is None else max_batches
    for batch_idx, batch in tqdm(enumerate(dataloader), total=tqdm_total):
        if max_batches is not None and batch_idx >= max_batches:
            break
        frames_a, frames_b, vec_a, vec_b, true_judgements = \
            batch['frames_a'], batch['frames_b'], \
            batch['vec_a'], batch['vec_b'], \
            batch['judgement']
        # Use model to predict probabilities of each judgement
        probs, pred_rewards = pref.predict_pref_probs(reward_model, frames_a, frames_b, vec_a, vec_b, ret_rewards=True)
        all_pred_rewards.append(pred_rewards)
        # Round probabilities to the closest judgement
        pred_judgements = pref.probs_to_judgements(probs.detach().numpy())
        assert pred_judgements.shape == true_judgements.shape
        # Compare prediction to true judgements
        eq = np.all(true_judgements.numpy() == pred_judgements, axis=1)
        correct_answers += eq.sum()
        total_answers += len(eq)
    mean_acc = correct_answers / total_answers
    if ret_rewards:
        all_pred_rewards = torch.cat(all_pred_rewards).flatten()
        return mean_acc, all_pred_rewards
    return mean_acc

def main(cfg):

    wandb.init(
        project=f"reward_training_{cfg.env_task}",
        # mode="disabled",
        tags=['basalt']
        )

    save_dir = Path(cfg.reward.save_dir) / wandb.run.name
    save_dir.mkdir(parents=True, exist_ok=True)

    utils.set_seeds(cfg.reward.rand_seed)

    reward_model = RewardModel()
    wandb.watch(reward_model)
    optimizer = torch.optim.Adam(
        reward_model.parameters(),
        lr=cfg.reward.adam_lr,
        betas=cfg.reward.adam_betas,
        eps=cfg.reward.adam_eps)

    full_dataset = TrajectoryPreferencesDataset(cfg.sampler.db_path)
    val_size = int(cfg.reward.val_split * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.reward.batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.reward.batch_size, shuffle=True, num_workers=0)

    samples_count = 0
    for epoch in range(cfg.reward.num_epochs):
        print("Epoch", epoch)
        for batch_idx, data_batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            if samples_count > cfg.reward.max_num_pairs:
                print(f"Hit max_num_pairs ({samples_count} / {cfg.reward.max_num_pairs}), quitting.")
                break

            frames_a, frames_b, vec_a, vec_b, judgements = \
                data_batch['frames_a'], data_batch['frames_b'], \
                data_batch['vec_a'], data_batch['vec_b'], \
                data_batch['judgement']

            # Run prediction pipeline
            prefer_probs, pred_rewards = pref.predict_pref_probs(
                reward_model, frames_a, frames_b, vec_a, vec_b, ret_rewards=True)
            assert pred_rewards.shape == (len(frames_a) * 2 * cfg.sampler.sample_length,),\
                (pred_rewards.shape, (len(frames_a) * 2 * cfg.sampler.sample_length,))

            # Calculate loss:
            # This is almost CrossEntropy but slightly different because we want to support
            # tie condition (0.5, 0.5) in judgement which is not possible in default XEnt
            # loss = - (judgement[0] * torch.log(prefer_1) + judgement[1] * torch.log(prefer_2))
            assert judgements.shape == prefer_probs.shape,\
                (judgements.shape, prefer_probs.shape)
            loss = - torch.sum(judgements * torch.log(prefer_probs))
            # An extra loss proportional to the square of the predicted rewards is added 
            # to impose a zero-mean Gaussian prior on the reward distribution.
            loss += cfg.reward.gaussian_prior_weight * torch.sum(pred_rewards.square())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluation & logging
            with torch.no_grad():
                reward_model.eval()
                wandb.log({"loss": loss.item()})
                train_acc = get_pred_accuracy(prefer_probs, judgements)
                wandb.log({"train_acc": train_acc})
                if batch_idx % cfg.reward.val_every_n_batch == 0:
                    # Validation accuracy
                    val_acc, val_rewards = evaluate_model_accuracy(
                        reward_model, val_dataloader, ret_rewards=True)
                    wandb.log({"val_rewards": wandb.Histogram(val_rewards)}) # For debugging
                    wandb.log({"val_acc": val_acc})
                reward_model.train()
            
            samples_count += len(frames_a)

            # Save model
            if batch_idx % cfg.reward.save_every_n_batch == 0:
                save_path = save_dir / f"{samples_count:06d}.pt"
                torch.save(reward_model.state_dict(), save_path)
                print("Saved model to", save_path)

        # TODO: A fraction of 1/e of the data is held out to be used as a validation set. 
        # We use L2- regularization of network weights with the adaptive scheme described in 
        # Christiano et al. (2017): the L2-regularization weight increases if the average 
        # validation loss is more than 50% higher than the average training loss, and decreases 
        # if it is less than 10% higher (initial weight 0.0001, multiplicative rate of change 
        # 0.001 per learning step).

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
    save_path = save_dir / f"{samples_count:06d}.pt"
    torch.save(reward_model.state_dict(), save_path)
    print("Saved model to", save_path)
    # Also save final model to the "best" path
    torch.save(reward_model.state_dict(), cfg.reward.best_model_path)
    print("Saved model to", cfg.reward.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train reward model with initial preferences')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    # Load config params
    cfg = utils.load_config(options.config_file)
    
    main(cfg)
