import h5py
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset, DataLoader

class H5Dataset(Dataset):
    def __init__(self, h5_file, target_ratio=0.5, subset_fraction=1.0, augment=False):
        self.h5_file = h5_file
        self.target_ratio = target_ratio
        self.subset_fraction = subset_fraction
        self.augment = augment
        self.transform = self._get_transforms() if augment else None
        self._balance_data()

    def _balance_data(self):
        with h5py.File(self.h5_file, 'r') as hdf:
            total_samples = len(hdf['images'])
            self.num_samples = int(total_samples * self.subset_fraction)
            
            # Select subset indices first and sort them
            subset_indices = np.sort(np.random.choice(np.arange(total_samples), self.num_samples, replace=False))
            
            labels = hdf['labels'][subset_indices]
            
            positive_indices = subset_indices[labels == 1]
            negative_indices = subset_indices[labels == 0]
            
            num_positive = len(positive_indices)
            num_negative = len(negative_indices)
            
            target_samples = int(self.num_samples * self.target_ratio)
            
            if num_positive < target_samples:
                positive_indices = np.random.choice(positive_indices, target_samples, replace=True)
            else:
                positive_indices = np.random.choice(positive_indices, target_samples, replace=False)
            
            if num_negative < target_samples:
                negative_indices = np.random.choice(negative_indices, target_samples, replace=True)
            else:
                negative_indices = np.random.choice(negative_indices, target_samples, replace=False)
            
            self.indices = np.sort(np.concatenate([positive_indices, negative_indices]))

    def _get_transforms(self):
        return A.Compose([
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as hdf:
            actual_idx = self.indices[idx]
            image = hdf['images'][actual_idx].astype(np.float32)
            label = hdf['labels'][actual_idx].astype(np.float32)
            
            if self.augment:
                augmented = self.transform(image=image)
                image = augmented['image']  #.permute(2, 0, 1)
            else:
                image = torch.from_numpy(image).permute(2, 0, 1)
            
            return image, torch.tensor(label)


class DataLoaderPyTorch:
    def __init__(self, train_file, batch_size=256, balance_data=True, num_workers=4,target_ratio=0.5 ,subset=1.0, augment=True):
        self.train_file = train_file
        self.batch_size = batch_size
        self.balance_data = balance_data
        self.num_workers = num_workers
        self.target_ratio = target_ratio
        self.augment = augment
        self.subset = subset
        self.train_loader = None
        self._prepare_data()
    def _prepare_data(self):
        if self.train_file:
            train_dataset = H5Dataset(self.train_file, target_ratio=0.5, subset_fraction=self.subset, augment=True)
            self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
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