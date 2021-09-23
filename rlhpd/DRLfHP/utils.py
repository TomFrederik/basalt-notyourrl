# import numpy as np

def simulate_judgement(clip_a, clip_b, diff_tol = 0.2):
    reward_a = sum([reward for (img, action, reward) in clip_a])
    reward_b = sum([reward for (img, action, reward) in clip_b])

    if abs(reward_a - reward_b) < diff_tol * max(abs(reward_a), abs(reward_b)):
        # Tie if the difference is less than tol% of the larger reward
        judgement = (0.5, 0.5)
    elif reward_a > reward_b:
        judgement = (1, 0)
    else: # reward_a < reward_b:
        judgement = (0, 1)
    return judgement