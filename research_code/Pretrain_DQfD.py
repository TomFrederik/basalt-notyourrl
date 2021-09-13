import argparse
import os
import random
from time import time

import einops
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from torch.utils.data import Dataset

from torch.utils.tensorboard import SummaryWriter

from DQfD_utils import MemoryDataset

def pretrain(log_dir, save_freq, dataset, discount_factor, q_net, pretrain_steps, batch_size, supervised_loss_margin, lr, weight_decay):
    
    writer = SummaryWriter(log_dir)
    
    optimizer = torch.optim.AdamW(q_net.parameters(), lr=lr, weight_decay=weight_decay)
    
    steps = 0
    while steps < pretrain_steps:
        steps += 1
        
        # get next batch
        batch_idcs = dataset.combined_memory.sample(batch_size)
        (pov, vec), (next_pov, next_vec), _, cur_expert_action, _, _, idcs, weight = zip([dataset[idx] for idx in batch_idcs])

        # forward pass
        cur_q_values = q_net.forward(...)
        with torch.no_grad():
            next_q_values = q_net.forward(...)
        
        # zero gradients
        optimizer.zero_grad(set_to_none=True)

        # compute loss
        pre_max_q = cur_q_values + supervised_loss_margin
        pre_max_q[np.arange(len(cur_expert_action)), cur_expert_action] -= supervised_loss_margin
        J_E = torch.max(pre_max_q, dim=1)[0] - cur_q_values[np.arange(len(cur_expert_action)), cur_expert_action]
        J_E = weight * J_E
        
        # backward and step
        J_E.backward()
        optimizer.step()
        
        # loss logging
        writer.add_scalar('Pretraining/J_E', J_E, global_step=steps)
        
        # sample weight updating with td_error
        # (reward is always 0 in pretraining)
        updated_td_errors = torch.abs(discount_factor * next_q_values - cur_q_values.detach())
        dataset.update_td_errors(batch_idcs, updated_td_errors)
        
        if steps % save_freq == 0:
            print('Saving model...')
            torch.save(q_net.state_dict(), os.path.join(log_dir, 'model.pt'))

    return q_net

def main(env_name, pretrain_steps, save_freq,
         lr, n_step, agent_memory_capacity, discount_factor, epsilon, batch_size, num_expert_episodes, data_dir, log_dir,
         PER_exponent, IS_exponent_0, agent_p_offset, expert_p_offset, weight_decay, supervised_loss_margin):
    
    # set save dir
    # TODO: save config in file path?
    log_dir = os.path.join(log_dir, env_name, str(int(time())))
    os.makedirs(log_dir, exist_ok=True)
    
    # set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # init q net
    q_net = None #TODO
    
    # init dataset
    p_offset=dict(expert=expert_p_offset, agent=agent_p_offset)
    dataset = MemoryDataset(
        agent_memory_capacity,
        n_step,
        discount_factor,
        p_offset,
        PER_exponent,
        IS_exponent_0,
        env_name,
        data_dir,
        num_expert_episodes
    )
    
    # launch pretraining
    q_net = pretrain(
        log_dir,
        save_freq,
        dataset,
        discount_factor, 
        q_net,
        pretrain_steps,
        batch_size,
        supervised_loss_margin,
        lr,
        weight_decay
    )
    
    print('Training finished! Saving model...')
    torch.save(q_net.state_dict(), os.path.join(log_dir, 'model.pt'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='MineRLTreechop-v0')
    parser.add_argument('--log_dir', default='/home/lieberummaas/datadisk/minerl/run_logs')
    parser.add_argument('--data_dir', default='/home/lieberummaas/datadisk/minerl/data')
    parser.add_argument('--num_expert_episodes', type=int, default=10)
    parser.add_argument('--n_step', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--action_repeat', type=int, default=5)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epsilon', type=float, default=0.01)
    parser.add_argument('--PER_exponent', type=float, default=0.4, help='PER exponent')
    parser.add_argument('--IS_exponent_0', type=float, default=0.6, help='Initial PER Importance Sampling exponent')
    parser.add_argument('--agent_p_offset', type=float, default=0.001)
    parser.add_argument('--expert_p_offset', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--supervised_loss_margin', type=float, default=0.8)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--agent_memory_capacity', type=int, default=20000)
    parser.add_argument('--pretrain_steps', type=int, default=10000)
    
    args = parser.parse_args()
    
    main(**vars(args))
