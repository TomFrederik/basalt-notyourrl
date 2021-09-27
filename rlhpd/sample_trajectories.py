import gym
import numpy as np
from database import return_all_ids, insert_traj_pair, create_table
import os
import random
import minerl

def step_random_policy(environment):
    action = environment.action_space.sample()
    observation, _, done, _ = environment.step(action)

    return observation, done

def generate_trajectory(max_traj_length,environment):
    done = False
    observation = environment.reset()

    trajectory = np.zeros(shape=(max_traj_length, 64,64,3))        
    step_idx = 0
    while not done and step_idx<max_traj_length:
            trajectory[step_idx] = observation["pov"]
            observation, done=step_random_policy(environment)
            step_idx += 1
    return trajectory

def save_trajectory(traj_name, trajectory, random_run_num):
    np.save(f"trajectories/{random_run_num}_traj_{traj_name}_full", trajectory)

def generate_sample(trajectory, max_traj_length, sample_length=60,):
    assert max_traj_length > sample_length
    rng = np.random.default_rng(12345)
    starting_idx =rng.integers(low=0, high = max_traj_length-sample_length)
    sample = trajectory[starting_idx:starting_idx+sample_length,...]
    return sample

def save_sample(traj_name,sample_name, sample, random_run_num):
    np.save(f"trajectories/{random_run_num}_traj_{traj_name}_smpl_{sample_name}", sample)

def fill_database(num_of_traj, num_of_samples, random_run_num,pair_per_sample=4):
    #names will be traj_x_smpl_y
    existing_ids=return_all_ids()
    if existing_ids:
        print("here1")
        for x in range(num_of_traj):
            for y in range(num_of_samples):
                for _ in range(pair_per_sample):
                    random_match = random.choice(existing_ids)[0]
                    print(random_match, random_run_num,x,y)
                    try: # if we picked a pair that exists we just skip for now TODO
                        insert_traj_pair(f"{random_run_num}_traj_{x}_smpl_{y}", random_match)
                    except:
                        continue
                        
    else: # no samples in database so far, add each id once to get started 
        for x1 in range(num_of_traj):

            x2 = np.mod(x1+1, num_of_traj)
            for y in range(num_of_samples):
                        print(x1,x2)
                        insert_traj_pair(f"{random_run_num}_traj_{x1}_smpl_{y}",f"{random_run_num}_traj_{x2}_smpl_{y}")
                
                


def main(max_traj_length=10, num_of_traj=3, num_of_samples=4, sample_length=5, pair_per_sample=2):
    environment = gym.make("MineRLObtainDiamond-v0")
    create_table()
    os.makedirs("trajectories", exist_ok=True)
    random_run_num = random.randint(100000,999999) # this is for getting new file names for each run
    print(random_run_num)
    for traj_idx in range(num_of_traj):
        trajectory=generate_trajectory(max_traj_length,environment)
        save_trajectory(traj_idx, trajectory, random_run_num)
        
        for sample_idx in range(num_of_samples):
            sample = generate_sample(trajectory, max_traj_length, sample_length )
            save_sample(traj_idx, sample_idx,sample, random_run_num)
            fill_database(num_of_traj,num_of_samples,random_run_num,pair_per_sample)



if __name__=="__main__":
    main()


    



