# import numpy as np

def simulate_judgement(clip_1, clip_2, diff_tol = 0.2):
    reward_1 = sum([reward for (img, action, reward) in clip_1])
    reward_2 = sum([reward for (img, action, reward) in clip_2])

    if abs(reward_1 - reward_2) < diff_tol * max(abs(reward_1), abs(reward_2)):
        # Tie if the difference is less than tol% of the larger reward
        judgement = (0.5, 0.5)
    elif reward_1 > reward_2:
        judgement = (1, 0)
    else: # reward_1 < reward_2:
        judgement = (0, 1)
    return judgement