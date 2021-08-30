from torch.utils.data import Dataset, IterableDataset
import minerl

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
        Returns next tuple in the iterator.
        '''
        obs, act, rew, next_obs, done = self.iter.buffered_batch_iter(self.batch_size, self.num_epochs)
        out = {
            'obs':obs,
            'acts':act,
            'next_obs':next_obs,
            'dones':done
        }
        yield out