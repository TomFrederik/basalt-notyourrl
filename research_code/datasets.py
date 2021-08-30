from torch.utils.data import Dataset, IterableDataset
import minerl

class VQVAEDataset(IterableDataset):
    '''
    For docs on BufferedBatchIter, see https://github.com/minerllabs/minerl/blob/dev/minerl/data/buffered_batch_iter.py
    '''
    def __init__(self, env_name, data_dir, batch_size):
        # save batch_size
        self.batch_size = batch_size
        
        # create data pipeline
        self.data = minerl.data.make(env_name, data_dir=data_dir)
        
        # create iterator from pipeline
        self.iter = minerl.data.BuffedBatchIter(self.data)       
         
    def __iter__(self):
        '''
        Returns next pov_obs in the iterator.
        '''
        return self.iter.buffed_batch_iter(self.batch_size)[0]['pov']


class AIRLDataset(IterableDataset):
    '''
    For docs on BufferedBatchIter, see https://github.com/minerllabs/minerl/blob/dev/minerl/data/buffered_batch_iter.py
    '''
    def __init__(self, env_name, data_dir, batch_size):
        # save batch_size
        self.batch_size = batch_size
        
        # create data pipeline
        self.data = minerl.data.make(env_name, data_dir=data_dir)
        
        # create iterator from pipeline
        self.iter = minerl.data.BuffedBatchIter(self.data)       
         
    def __iter__(self):
        '''
        Returns next tuple in the iterator.
        '''
        obs, act, rew, next_obs, done = self.iter.buffed_batch_iter(self.batch_size)
        out = {
            'obs':obs,
            'acts':act,
            'next_obs':next_obs,
            'dones':done
        }
        yield out