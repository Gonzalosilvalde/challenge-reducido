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

def load_model(model_path, device):
    print(f"Loading model from {model_path}")
    model = timm.create_model("resnet18",in_chans=6, num_classes=10)
    model = train.modify_resnet18(model, num_input_channels=6)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    return model

@torch.no_grad()
def predict(model, test_loader, device):
    print("Starting prediction process")
    predictions = []
    ids = []
    
    for inputs, batch_ids in tqdm(test_loader, desc="Predicting", unit="batch"):
        inputs = inputs.to(device, non_blocking=True)
        
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(inputs)
            probs = (torch.sigmoid(outputs.squeeze(1))).float() #> 0.5).float()
        
        predictions.extend(probs.cpu().numpy().tolist())  # Convert to Python list
        ids.extend(batch_ids.cpu().numpy().tolist())  # Convert to Python list
    
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
    
    # Create a dictionary for faster lookup, reversing the mapping
    id_dict = dict(zip(id_map['ID'].astype(str), id_map['id']))
    
    # Map the numeric IDs to their corresponding string IDs
    predictions['id'] = predictions['id'].astype(str).map(id_dict)
    
    predictions.to_csv(output_file, index=False)
    print(f"Submission file created: {output_file}")


def main():
    print("Starting main function")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = "checkpoints/best_model.pth"
    test_file = "data/test_data.h5"
    batch_size = 4096
    id_map_file = 'id_map.csv'
    sample_submission_file = 'SampleSubmission.csv'
    output_file = 'submission6.csv'
    predictions_file = 'predictions.csv'

    model = load_model(model_path, device)
    print("Creating test dataloader")

    submit_data = submit_loader.SubmitTestLoader(test_file, batch_size=batch_size, subset_fraction=1.0)
    test_loader = submit_data.get_test_loader()
    
    #print(f"Test loader created with {len(submit_data)} samples")
    
    predictions, ids = predict(model, test_loader, device)

    
    #save_predictions_to_json(predictions, ids, predictions_file)
    save_predictions_to_csv (predictions, ids, predictions_file)
    create_submission_file(predictions_file, id_map_file, output_file)
    
    print("Script execution completed")
    print("\nTo submit your results, please visit:")
    print("https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge/submit")

if __name__ == "__main__":
    main()