import h5py
import numpy as np
from tqdm import tqdm

def check_dataset_balance(file_path):
    with h5py.File(file_path, 'r') as f:
        labels = f['labels']
        total_samples = len(labels)
        
        zeros = 0
        ones = 0
        
        for label in tqdm(labels, desc="Counting labels"):
            if label == 0:
                zeros += 1
            elif label == 1:
                ones += 1
        
        print(f"Total samples: {total_samples}")
        print(f"Number of 0s: {zeros} ({zeros/total_samples*100:.2f}%)")
        print(f"Number of 1s: {ones} ({ones/total_samples*100:.2f}%)")

if __name__ == "__main__":
    train_file = "data/train_data.h5"
    check_dataset_balance(train_file)