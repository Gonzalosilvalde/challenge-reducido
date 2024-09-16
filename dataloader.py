import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from functools import lru_cache
from torchvision import transforms

class H5Dataset(Dataset):
    def __init__(self, h5_file, balance_data=False, subset_fraction=1.0, augment=False):
        self.h5_file = h5_file
        self.balance_data = balance_data
        self.subset_fraction = subset_fraction
        self.augment = augment
        self.transform = self._get_transforms() if augment else None
        
        with h5py.File(h5_file, 'r') as hdf:
            total_samples = len(hdf['images'])
            self.num_samples = int(total_samples * subset_fraction)
            if self.balance_data and 'labels' in hdf:
                self.indices = self._balance_data(hdf)
            else:
                self.indices = np.random.choice(total_samples, self.num_samples, replace=False)

    @lru_cache(maxsize=1)
    def _balance_data(self, hdf):
        y_data = hdf['labels'][:self.num_samples]
        num_ones = int((y_data == 1).sum() * self.subset_fraction)
        ones_indices = np.where(y_data == 1)[0][:num_ones]
        zeros_indices = np.where(y_data == 0)[0]
        balanced_zero_indices = np.random.choice(zeros_indices, num_ones, replace=False)
        return np.concatenate([ones_indices, balanced_zero_indices])

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Lambda(self._custom_transform)
        ])

    def _custom_transform(self, img):
        
        # Dividimos los 6 canales en dos grupos de 3 canales
        img1, img2 = img[:3], img[3:]

        # Aplicamos las transformaciones a cada grupo por separado
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])

        img1 = transform(img1)
        img2 = transform(img2)

        # Recombinamos los dos grupos
        return torch.cat([img1, img2], dim=0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        with h5py.File(self.h5_file, 'r', libver='latest', swmr=True) as hdf:
            X = torch.tensor(hdf['images'][actual_idx], dtype=torch.float32).permute(2, 0, 1)
            if self.augment:
                X = self.transform(X)
            if 'labels' in hdf:
                y = torch.tensor(hdf['labels'][actual_idx], dtype=torch.float32)
                return X, y
            else: 
                return X

import h5py
import torch
from torch.utils.data import Dataset

class TestH5Dataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as hdf:
            self.num_samples = len(hdf['images'])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r', libver='latest', swmr=True) as hdf:
            X = torch.tensor(hdf['images'][idx], dtype=torch.float32).permute(2, 0, 1)
            id = self.get_id(idx)
            return X, id

    def get_id(self, idx):
        with h5py.File(self.h5_file, 'r', libver='latest', swmr=True) as hdf:
            if 'ids' in hdf:
                return hdf['ids'][idx]
            else:
                return idx  

class DataLoaderPyTorch:
    def __init__(self, train_file, test_file=None, batch_size=256, balance_data=True, num_workers=4, train_subset=1.0, test_subset=1.0, augment_train=True):
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.balance_data = balance_data
        self.num_workers = num_workers
        self.train_subset = train_subset
        self.test_subset = test_subset
        self.augment_train = augment_train

        self.train_loader = None
        self.test_loader = None

        self._prepare_data()

    def _prepare_data(self):
        if self.train_file:
            train_dataset = H5Dataset(self.train_file, balance_data=self.balance_data, subset_fraction=self.train_subset, augment=self.augment_train)
            self.train_loader = self._create_dataloader(train_dataset, shuffle=True)

        if self.test_file:
            test_dataset = H5Dataset(self.test_file, balance_data=False, subset_fraction=self.test_subset, augment=False)
            self.test_loader = self._create_dataloader(test_dataset, shuffle=False)

    def _create_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=self.num_workers, 
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

    def _verify_batch_shapes(self):
        self._print_batch_shapes("Train loader", self.train_loader)
        if self.test_loader:
            self._print_batch_shapes("Test loader", self.test_loader)

    def _print_batch_shapes(self, loader_name, loader):
        print(f"{loader_name} batch shapes:")
        for batch in loader:
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

# Example usage
# data_loader = DataLoaderPyTorch(train_file="train_data.h5", test_file="test_data.h5", train_subset=0.8, test_subset=0.5, augment_train=True)
# train_loader = data_loader.get_train_loader()
# test_loader = data_loader.get_test_loader()