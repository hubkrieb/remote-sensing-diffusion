import os
import torch
from torch.utils import data
import csv
import lmdb
import torch

class SegmentationDataset(data.Dataset):
        def __init__(self, db_path, split_path, split, n_channels = 3):

            self.db_path = db_path

            self.split = split
            self.n_channels = n_channels
            with open(f'{split_path}/{split}.csv') as f:
                reader = csv.reader(f)
                self.filenames = list(reader)
            self.env = None
            self.txn = None
            self.image_tensor = torch.ones((3, 256, 256))
            self.mask_tensor = torch.ones((1, 256, 256))
        
        def _init_db(self):
            self.env = lmdb.open(self.db_path, subdir=os.path.isdir(self.db_path),
                readonly=True, lock=False,
                readahead=False, meminit=False)
            self.txn = self.env.begin()

        def read_lmdb_image(self, key):
            lmdb_data = self.txn.get(key.encode('ascii'))
            lmdb_data = torch.frombuffer(lmdb_data, dtype = torch.float32)
            lmdb_data = lmdb_data.reshape((3, 256, 256))
            return lmdb_data
        
        def read_lmdb_mask(self, key):
            lmdb_data = self.txn.get(key.encode('ascii'))
            lmdb_data = torch.frombuffer(lmdb_data, dtype = torch.float32)   
            lmdb_data = lmdb_data.reshape((1, 256, 256))
            return lmdb_data
        
        def __len__(self):
            return len(self.filenames)
        
        def __getitem__(self, idx):
            if self.env is None:
                 self._init_db()
            filename = self.filenames[idx][0]
            image = self.read_lmdb_image(filename)
            mask = self.read_lmdb_mask(filename + '_mask')
            return image, mask

def get_dataloader(db_path, n_channels, batch_size, split_path, split, shuffle, num_workers = 0):
    """ Generates Training, Validation and Test Dataloaders"""

    dataset = SegmentationDataset(db_path, split_path, split, n_channels)
    dataloader = data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    return dataloader