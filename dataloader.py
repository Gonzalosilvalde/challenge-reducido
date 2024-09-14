import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache

class H5Dataset(Dataset):
    def __init__(self, h5_file, balance_data=False):
        self.h5_file = h5_file
        self.balance_data = balance_data
        
        with h5py.File(h5_file, 'r') as hdf:
            self.num_samples = len(hdf['images'])
            if self.balance_data and 'labels' in hdf:
                self.indices = self._balance_data(hdf)
            else:
                self.indices = np.arange(self.num_samples)

    @lru_cache(maxsize=1)
    def _balance_data(self, hdf):
        y_data = hdf['labels'][:]
        num_ones = (y_data == 1).sum()
        ones_indices = np.where(y_data == 1)[0]
        zeros_indices = np.where(y_data == 0)[0]
        balanced_zero_indices = np.random.choice(zeros_indices, num_ones, replace=False)
        return np.concatenate([ones_indices, balanced_zero_indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        with h5py.File(self.h5_file, 'r') as hdf:
            X = hdf['images'][actual_idx]
            if 'labels' in hdf:
                y = hdf['labels'][actual_idx]
                return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
            else:
                return torch.tensor(X, dtype=torch.float32)

class DataLoaderPyTorch:
    def __init__(self, train_file, test_file, batch_size=256, balance_data=True, num_workers=4):
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.balance_data = balance_data
        self.num_workers = num_workers

        self.train_loader = None
        self.test_loader = None

        self._prepare_data()

    def _prepare_data(self):
        train_dataset = H5Dataset(self.train_file, balance_data=self.balance_data)
        test_dataset = H5Dataset(self.test_file, balance_data=False)

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            pin_memory=True,
            prefetch_factor=2
        )
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True,
            prefetch_factor=2
        )

        self._verify_batch_shapes()

    def _verify_batch_shapes(self):
        print("Train loader batch shapes:")
        for batch in self.train_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:  
                inputs, labels = batch
                print(f'Batch input shape: {inputs.shape}')
                print(f'Batch labels shape: {labels.shape}')
            else:
                print(f'Batch shape: {batch.shape}')
            break

        print("Test loader batch shapes:")
        for batch in self.test_loader:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                inputs, labels = batch
                print(f'Batch input shape: {inputs.shape}')
                print(f'Batch labels shape: {labels.shape}')
            else:
                print(f'Batch shape: {batch.shape}')
            break

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

"""# Ejemplo de uso
data_loader = DataLoaderPyTorch(train_file="train_data.h5", test_file="test_data.h5")
train_loader = data_loader.get_train_loader()
test_loader = data_loader.get_test_loader()
"""