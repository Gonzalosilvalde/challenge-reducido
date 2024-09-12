import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class H5Dataset(Dataset):
    def __init__(self, h5_file, balance_data=False):
        self.h5_file = h5_file
        self.balance_data = balance_data
        
        with h5py.File(h5_file, 'r') as hdf:
            self.X_data = hdf['images']
            self.y_data = hdf['labels'] if 'labels' in hdf else None

            if self.balance_data and self.y_data is not None:
                self._balance_data(hdf)
            else:
                self.indices = np.arange(len(self.X_data))  # Usar todos los índices si no hay balanceo

    def _balance_data(self, hdf):
        y_data = hdf['labels'][:]
        num_ones = (y_data == 1).sum()
        ones_indices = np.where(y_data == 1)[0]
        zeros_indices = np.where(y_data == 0)[0]
        balanced_zero_indices = np.random.choice(zeros_indices, num_ones, replace=False)
        self.indices = np.concatenate([ones_indices, balanced_zero_indices])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        # Abrimos el archivo .h5 en cada llamada para obtener el dato específico bajo demanda
        with h5py.File(self.h5_file, 'r') as hdf:
            X = hdf['images'][actual_idx]
            if self.y_data is not None:
                y = hdf['labels'][actual_idx]
                return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
            else:
                return torch.tensor(X, dtype=torch.float32)

class DataLoaderPyTorch:
    def __init__(self, train_file, test_file, batch_size=64, balance_data=True):
        self.train_file = train_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.balance_data = balance_data

        self.train_loader = None
        self.test_loader = None

        self._prepare_data()

    def _prepare_data(self):
        # Crear los datasets con lectura bajo demanda
        train_dataset = H5Dataset(self.train_file, balance_data=self.balance_data)
        test_dataset = H5Dataset(self.test_file, balance_data=False)

        # Crear los DataLoaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Verificar tamaños de lotes
        self._verify_batch_shapes()

    def _verify_batch_shapes(self):
        # Verificar el primer lote de los datos de entrenamiento
        print("Train loader batch shapes:")
        for batch in self.train_loader:
            if len(batch) == 2:  
                inputs, labels = batch
                print(f'Batch input shape: {inputs.shape}')
                print(f'Batch labels shape: {labels.shape}')
            break

        # Verificar el primer lote de los datos de prueba
        print("Test loader batch shapes:")
        for batch in self.test_loader:
            inputs = batch
            print(f'Batch input shape: {inputs.shape}')
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