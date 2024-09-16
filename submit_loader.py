import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class SubmitTestDataset(Dataset):
    def __init__(self, h5_file, subset_fraction=1.0):
        self.h5_file = h5_file
        self.subset_fraction = subset_fraction
        
        with h5py.File(h5_file, 'r') as hdf:
            total_samples = len(hdf['images'])
            self.num_samples = int(total_samples * subset_fraction)
            self.indices = torch.randperm(total_samples)[:self.num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        with h5py.File(self.h5_file, 'r', libver='latest', swmr=True) as hdf:
            X = torch.tensor(hdf['images'][actual_idx], dtype=torch.float32).permute(2, 0, 1)
            id = self.get_id(actual_idx)
            return X, id

    def get_id(self, idx):
        with h5py.File(self.h5_file, 'r', libver='latest', swmr=True) as hdf:
            if 'ids' in hdf:
                return hdf['ids'][idx]
            else:
                return idx

class SubmitTestLoader:
    def __init__(self, test_file, batch_size=256, num_workers=4, subset_fraction=1.0):
        self.test_file = test_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subset_fraction = subset_fraction

        self.test_loader = self._prepare_data()

    def _prepare_data(self):
        test_dataset = SubmitTestDataset(self.test_file, subset_fraction=self.subset_fraction)
        return DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True
        )

    def get_test_loader(self):
        return self.test_loader