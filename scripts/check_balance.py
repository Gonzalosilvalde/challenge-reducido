import h5py
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def split_and_analyze_dataset(file_path, validation_size=0.2, random_state=42):
    with h5py.File(file_path, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]

    # Split the dataset
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=validation_size, random_state=random_state, stratify=labels
    )

    # Save the split datasets
    with h5py.File('data/train_split.h5', 'w') as f:
        f.create_dataset('images', data=train_images)
        f.create_dataset('labels', data=train_labels)

    with h5py.File('data/validation_split.h5', 'w') as f:
        f.create_dataset('images', data=val_images)
        f.create_dataset('labels', data=val_labels)

    # Analyze class balance
    print("Analyzing class balance...")
    analyze_class_balance("Training set", train_labels)
    analyze_class_balance("Validation set", val_labels)

def analyze_class_balance(set_name, labels):
    total_samples = len(labels)
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = dict(zip(unique, counts))

    print(f"\n{set_name}:")
    print(f"Total samples: {total_samples}")
    for label, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"Class {label}: {count} samples ({percentage:.2f}%)")

if __name__ == "__main__":
    train_file = "data/train_data.h5"
    val_file_split = "data/validation_split.h5"
    train_file_split = "data/train_split.h5"

    #split_and_analyze_dataset(train_file)
    with h5py.File(val_file_split, 'r') as f:
        #images_val = f['images'][:]
        labels_val = f['labels'][:]
    with h5py.File(train_file_split, 'r') as f:
        #images_train = f['images'][:]
        labels_train = f['labels'][:]
    analyze_class_balance("Training set", labels_train)
    analyze_class_balance("Validation set", labels_val)