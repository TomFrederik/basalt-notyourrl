import einops
import numpy as np
import torch


def simulate_judgement(clip_a, clip_b, diff_tol = 0.2):
    reward_a = sum([reward for (state, action, reward, next_state, done, meta) in clip_a])
    reward_b = sum([reward for (state, action, reward, next_state, done, meta) in clip_b])

    if abs(reward_a - reward_b) < diff_tol * max(abs(reward_a), abs(reward_b)):
        # Tie if the difference is less than tol% of the larger reward
        judgement = (0.5, 0.5)
    elif reward_a > reward_b:
        judgement = (1, 0)
    else: # reward_a < reward_b:
        judgement = (0, 1)
    return judgement

def probs_to_judgements(probs_array):
    assert probs_array.shape == (len(probs_array), 2)
    judgements = np.zeros_like(probs_array)
    # probs_array[probs_array <= 0.33] = 0
    judgements[np.logical_and(0.33 < probs_array, probs_array <= 0.66)] = 0.5
    judgements[0.66 < probs_array] = 1
    return judgements

def predict_pref_probs(reward_model, frames_a, frames_b, vec_a, vec_b, ret_rewards=False):
    clip_length = frames_a.shape[1] # shape is (b, t, c, w, h)
    vec_length = vec_a.shape[2] # shape is (b, t, v)
    current_batch_size = len(frames_a) # == cfg.batch_size except at the end of the dataset

    batch_imgs = torch.stack([frames_a, frames_b], axis=1)
    batch_vecs = torch.stack([vec_a, vec_b], axis=1)
    assert batch_imgs.shape == (current_batch_size, 2, clip_length, 3, 64, 64)
    assert batch_vecs.shape == (current_batch_size, 2, clip_length, vec_length)
    
    # Flatten the (batch, Trajectory, time) dimensions to feed B,C,W,H into CNN
    batch_imgs = einops.rearrange(batch_imgs, 'b T t c w h -> (b T t) c w h')
    batch_vecs = einops.rearrange(batch_vecs, 'b T t v -> (b T t) v')
    assert batch_imgs.shape == (current_batch_size * 2 * clip_length, 3, 64, 64)
    assert batch_vecs.shape == (current_batch_size * 2 * clip_length, vec_length)

    # Predict reward of every frame in our batch
    r_preds = reward_model(
        batch_imgs,
        batch_vecs,
        ).squeeze()
    assert r_preds.shape == (len(batch_imgs),)

    # Reshape and sum up rewards along each
    all_rewards = r_preds.view(current_batch_size, 2, clip_length)
    assert all_rewards.shape == (current_batch_size, 2, clip_length)
    reward_sums = torch.sum(all_rewards, dim=2)
    assert reward_sums.shape == (current_batch_size, 2)

    # Convert pairs of rewards into preference probabilities
    prefer_probs = torch.softmax(reward_sums, dim=1)
    assert prefer_probs.shape == (current_batch_size, 2)
    if ret_rewards:
        # Rewards vector is useful during training for regularization loss
        return prefer_probs, r_preds
    return prefer_probs
