import torch
import h5py
import pandas as pd
from torch.amp import autocast
from net import Net  # Assuming your model class is in a file named net.py
from torch.utils.data import DataLoader
from dataloader import DataLoaderPyTorch  # Assuming you have this custom dataloader
from tqdm import tqdm
import timm
import train

def load_model(model_path, device):
    print(f"Loading model from {model_path}")
    #model = Net().to(device)
    #checkpoint = torch.load(model_path, map_location=device)
    model = timm.create_model("resnet18", num_classes=10)
    checkpoint = torch.load(model_path, map_location=device)
    model = train.modify_resnet18(model, num_input_channels=6)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded successfully")
    return model

def predict(model, test_loader, device):
    print("Starting prediction process")
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Predicting", unit="batch"):
            inputs = inputs.to(device)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(inputs)
                probs = torch.sigmoid(outputs.squeeze(1))
            
            preds = probs.float()
            
            predictions.extend(preds.cpu().numpy())
    
    print(f"Prediction complete. Total predictions: {len(predictions)}")
    return predictions

def create_submission_file(predictions, id_map_file, sample_submission_file, output_file):
    print("Creating submission file")
    # Load ID mapping
    df_id_map = pd.read_csv(id_map_file)
    id_map = {row['id']: row['ID'] for _, row in df_id_map.iterrows()}
    print(f"Loaded ID map with {len(id_map)} entries")
    
    # Load sample submission file
    sub = pd.read_csv(sample_submission_file)
    print(f"Loaded sample submission file with {len(sub)} entries")
    
    # Update predictions
    for i in tqdm(range(len(sub)), desc="Updating predictions", unit="entry"):
        sub.at[i, 'class'] = predictions[id_map[sub.at[i, 'id']]]
    
    # Save updated submission file
    sub.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

def main():
    print("Starting main function")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = "checkpoints/best_model2.pth"  # Adjust this path as needed
    test_file = "data/test_data.h5"
    batch_size = 2048  # Adjust as needed
    id_map_file = 'id_map.csv'
    sample_submission_file = 'SampleSubmission.csv'
    output_file = 'submission.csv'
    
    # Load the model
    model = load_model(model_path, device)
    
    # Create test dataloader
    print("Creating test dataloader")
    data_loader = DataLoaderPyTorch(None, test_file=test_file, batch_size=batch_size, train_subset=0.0, test_subset=1.0)
    test_loader = data_loader.get_test_loader()
    test_dataset = test_loader.dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test loader created with {len(test_dataset)} samples")

    # Make predictions
    predictions = predict(model, test_loader, device)
    
    # Create submission file
    create_submission_file(predictions, id_map_file, sample_submission_file, output_file)
    
    print("Script execution completed")
    print("\nTo submit your results, please visit:")
    print("https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge/submit")

if __name__ == "__main__":
    main()