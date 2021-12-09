from torch.utils.data import Dataset, IterableDataset
import minerl
import itertools
import einops
import numpy as np

class VQVAEDataset(IterableDataset):
    '''
    For docs on BufferedBatchIter, see https://github.com/minerllabs/minerl/blob/dev/minerl/data/buffered_batch_iter.py
    '''
    def __init__(self, env_name, data_dir, batch_size, num_epochs):
        # save params
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # create data pipeline
        self.data = minerl.data.make(env_name, data_dir=data_dir)
        
        # create iterator from pipeline
        self.iter = minerl.data.BufferedBatchIter(self.data)       
         
    def __iter__(self):
        '''
        Returns next pov_obs in the iterator.
        '''
        return self.iter.buffered_batch_iter(self.batch_size, self.num_epochs)


class AIRLDataset(IterableDataset):
    '''
    For docs on BufferedBatchIter, see https://github.com/minerllabs/minerl/blob/dev/minerl/data/buffered_batch_iter.py
    This Dataset matches the expectations of AdversarialTrainer
    '''
    def __init__(self, env_name, data_dir, batch_size, num_epochs):
        # save params
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        
        # create data pipeline
        self.data = minerl.data.make(env_name, data_dir=data_dir)
        
        # create iterator from pipeline
        self.iter = minerl.data.BufferedBatchIter(self.data)       
        self.iter = self.iter.buffered_batch_iter(self.batch_size, self.num_epochs)
        
    def get_generator(self, obs, act, rew, next_obs, done):
        '''
        Returns next tuple in the iterator.
        '''
        obs = einops.rearrange(obs['pov'].astype(np.float32) / 255, 'b h w c -> b c h w')
        next_obs = einops.rearrange(next_obs['pov'].astype(np.float32) / 255, 'b h w c -> b c h w')
        out = {
            'obs':obs,
            'acts':act['vector'],
            'next_obs':next_obs,
            'dones':done
        }
        return out
    
    def __iter__(self):
        return itertools.starmap(self.get_generator, self.iter)