import torch
import h5py
import json
import pandas as pd
from torch.amp import autocast
from torch.utils.data import DataLoader
import dataloader
from tqdm import tqdm
import timm
import train
import torch.cuda.amp
import submit_loader
import xgboost as xgb
import numpy as np

def load_model(model_path, device):
    print(f"Loading model from {model_path}")
    if model_path.endswith('.pth'):
        model = timm.create_model("resnet18", in_chans=6, num_classes=1)
        model = train.modify_resnet18(model, num_input_channels=6)
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model = model.to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("PyTorch model loaded successfully")
    elif model_path.endswith('.json'):
        model = xgb.Booster()
        model.load_model(model_path)
        print("XGBoost model loaded successfully")
    else:
        raise ValueError("Unsupported model file format. Use .pth for PyTorch or .json for XGBoost.")
    return model

def preprocess_for_xgboost(inputs):
    # Flatten 16x16x6 to 1536
    inputs_flattened = inputs.reshape(inputs.shape[0], -1)
    # Normalize to [0, 1]
    inputs_normalized = inputs_flattened.astype(np.float32) / 255.0
    return inputs_normalized

@torch.no_grad()
def predict(model, test_loader, device, model_type):
    print("Starting prediction process")
    predictions = []
    ids = []
    
    for inputs, batch_ids in tqdm(test_loader, desc="Predicting", unit="batch"):
        if model_type == 'pytorch':
            inputs = inputs.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                probs = (torch.sigmoid(outputs.squeeze(1))).float()
            predictions.extend(probs.cpu().numpy().tolist())
        elif model_type == 'xgboost':
            inputs = inputs.numpy()
            inputs_preprocessed = preprocess_for_xgboost(inputs)
            dmatrix = xgb.DMatrix(inputs_preprocessed)
            probs = model.predict(dmatrix)
            predictions.extend(probs.tolist())
        
        ids.extend(batch_ids.cpu().numpy().tolist())
    
    print(f"Prediction complete. Total predictions: {len(predictions)}")
    return predictions, ids

def save_predictions_to_json(predictions, ids, output_file):
    print(f"Saving predictions to {output_file}")
    data = {"ids": ids, "predictions": predictions}
    with open(output_file, 'w') as f:
        json.dump(data, f)
    print(f"Predictions saved to {output_file}")

def save_predictions_to_csv(predictions, ids, output_file):
    print(f"Saving predictions to {output_file}")
    df = pd.DataFrame({'id': ids, 'class': predictions})
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def create_submission_file(predictions_file, map_file, output_file):
    predictions = pd.read_csv(predictions_file)
    id_map = pd.read_csv(map_file)
    
    id_dict = dict(zip(id_map['ID'].astype(str), id_map['id']))
    predictions['id'] = predictions['id'].astype(str).map(id_dict)
    
    predictions.to_csv(output_file, index=False)
    print(f"Submission file created: {output_file}")

def main():
    print("Starting main function")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = "checkpoints/xgboost_model.json" #"checkpoints/best_model.pth" 
    test_file = "data/test_data.h5"
    batch_size = 256
    id_map_file = 'submissions/id_map.csv'
    output_file = 'submission10.csv'
    predictions_file = 'predictions.csv'

    model = load_model(model_path, device)
    model_type = 'pytorch' if model_path.endswith('.pth') else 'xgboost'
    
    print("Creating test dataloader")
    submit_data = submit_loader.SubmitTestLoader(test_file, batch_size=batch_size, subset_fraction=1.0)
    test_loader = submit_data.get_test_loader()
    print(f"Test loader created with {len(test_loader.dataset)} samples")
    
    predictions, ids = predict(model, test_loader, device, model_type)
    
    save_predictions_to_json(predictions, ids, 'predictions.json')
    save_predictions_to_csv(predictions, ids, predictions_file)
    create_submission_file(predictions_file, id_map_file, output_file)
    
    print("Script execution completed")
    print("\nTo submit your results, please visit:")
    print("https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge/submit")

if __name__ == "__main__":
    main()