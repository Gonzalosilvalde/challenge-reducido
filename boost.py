import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import h5py
from tqdm import tqdm

def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        X = f['images'][:]
        y = f['labels'][:]
    return X, y

def preprocess_data(X, y):
    X = X.reshape(X.shape[0], -1)  # Flatten 16x16x6 to 1536
    X = X.astype(np.float32) / 255.0  # Normalize to [0, 1]
    y = y.astype(np.float32)  # XGBoost works with float labels
    return X, y

def train_xgboost_gpu(X_train, y_train, X_val, y_val):
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set XGBoost parameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),  # Handle class imbalance
    }

    # Train XGBoost model
    num_round = 1000
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    
    print("Training XGBoost model...")
    bst = xgb.train(params, dtrain, num_round, evals=evallist, early_stopping_rounds=10, verbose_eval=10)
    
    return bst

def evaluate_model(model, X, y, dataset_name):
    dmatrix = xgb.DMatrix(X)
    y_pred = model.predict(dmatrix)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    accuracy = accuracy_score(y, y_pred_binary)
    precision = precision_score(y, y_pred_binary)
    recall = recall_score(y, y_pred_binary)
    f1 = f1_score(y, y_pred_binary)
    
    print(f"\n{dataset_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def main():
    # Load and preprocess data
    train_file = "data/train_split.h5"
    validation_file = "data/validation_split.h5"

    print("Loading and preprocessing data...")
    X_train, y_train = load_data(train_file)
    X_val, y_val = load_data(validation_file)

    X_train, y_train = preprocess_data(X_train, y_train)
    X_val, y_val = preprocess_data(X_val, y_val)

    # Train XGBoost model
    model = train_xgboost_gpu(X_train, y_train, X_val, y_val)

    # Evaluate the model
    evaluate_model(model, X_train, y_train, "Training")
    evaluate_model(model, X_val, y_val, "Validation")

    # Save the model
    model.save_model('xgboost_model.json')
    print("\nModel saved as 'xgboost_model.json'")

if __name__ == "__main__":
    main()