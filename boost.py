from concurrent.futures import ThreadPoolExecutor
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import h5py
import os
from tqdm import tqdm
import wandb
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        X = f['images'][:]
        y = f['labels'][:]
    return X, y

def save_processed_data(X, y, file_path):
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('features', data=X)
        f.create_dataset('labels', data=y)

def load_processed_data(file_path):
    with h5py.File(file_path, 'r') as f:
        X = f['features'][:]
        y = f['labels'][:]
    return X, y


def process_sample(X, i):
    sample_features = []
    # Spectral indices
    nir, red = X[i, :, :, 3], X[i, :, :, 2]
    ndvi = (nir - red) / (nir + red + 1e-8)  # Normalized Difference Vegetation Index
    swir1, green = X[i, :, :, 4], X[i, :, :, 1]
    ndwi = (green - swir1) / (green + swir1 + 1e-8)  # Normalized Difference Water Index
    sample_features.extend([ndvi.mean(), ndvi.std(), ndwi.mean(), ndwi.std()])
    
    for j in range(X.shape[3]):
        band = X[i, :, :, j]
        # Basic statistics
        sample_features.extend([
            band.mean(), band.std(), band.min(), band.max()
        ])
        # Texture features (GLCM)
        band_uint8 = (band * 255).astype(np.uint8)
        glcm = graycomatrix(band_uint8, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, normed=True)
        sample_features.extend([
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'dissimilarity').mean(),
            graycoprops(glcm, 'homogeneity').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'correlation').mean()
        ])
    return sample_features

def preprocess_data(X, y):
    n_samples = X.shape[0]
    X = X.astype(np.float32) / 65535.0  # Normalize to [0, 1] (Landsat 16-bit data)
    
    with ThreadPoolExecutor() as executor:
        features = list(tqdm(executor.map(lambda i: process_sample(X, i), range(n_samples)), total=n_samples, desc="Processing samples"))
    
    X_features = np.array(features)
    print(f"X_features shape: {X_features.shape}")
    # Standardize features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    print("Features standardized")
    # Convert labels to integers
    y = y.astype(np.int32)
    
    return X_features, y

def train_xgboost_gpu(X_train, y_train, X_val, y_val):
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set XGBoost parameters
    params = {
        'max_depth': 9,
        'learning_rate': 0.01,
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist',
        'device': 'cuda',
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),  # Handle class imbalance
    }

    # Train XGBoost model
    num_round = 1000
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    print("Training XGBoost model...")
    bst = xgb.train(params, dtrain, num_round, evals=evallist, early_stopping_rounds=50,
                    callbacks=[wandb.xgboost.WandbCallback(log_model=True)])
    
    return bst

def evaluate_model(model, X, y, dataset_name):
    dmatrix = xgb.DMatrix(X)
    y_pred = model.predict(dmatrix)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    f1 = f1_score(y, y_pred_binary)
    auc = roc_auc_score(y, y_pred)
    
    metrics = {
        f'{dataset_name}/Accuracy': accuracy,
        f'{dataset_name}/Precision': precision,
        f'{dataset_name}/Recall': recall,
        f'{dataset_name}/F1': f1,
        f'{dataset_name}/AUC': auc
    }
    
    wandb.log(metrics)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

def main():
    # Initialize wandb
    wandb.init(project="landsat-classification", config={
        "model": "XGBoost",
        "dataset": "Landsat 16x16 patches"
    })

    # Load and preprocess data
    train_file = "data/train_split.h5"
    validation_file = "data/validation_split.h5"
    processed_train_file = "data/processed_train.h5"
    processed_val_file = "data/processed_val.h5"

    print("Loading and preprocessing data...")
    if os.path.exists(processed_train_file) and os.path.exists(processed_val_file):
        print("Loading preprocessed data...")
        X_train, y_train = load_processed_data(processed_train_file)
        X_val, y_val = load_processed_data(processed_val_file)
    else:
        print("Preprocessing data...")
        X_train, y_train = load_data(train_file)
        X_val, y_val = load_data(validation_file)

        X_train, y_train = preprocess_data(X_train, y_train)
        X_val, y_val = preprocess_data(X_val, y_val)

        print("Saving preprocessed data...")
        save_processed_data(X_train, y_train, processed_train_file)
        save_processed_data(X_val, y_val, processed_val_file)

    # Train XGBoost model
    with tqdm(total=100, desc="Training Progress") as pbar:
        def callback(env):
            pbar.update(1)
        model = train_xgboost_gpu(X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(model, X_train, y_train, "Training")
    evaluate_model(model, X_val, y_val, "Validation")

    # Save the model
    model.save_model('xgboost_model.json')
    wandb.save('xgboost_model.json')
    print("\nModel saved as 'xgboost_model.json'")

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()