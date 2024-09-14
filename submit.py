import torch
import h5py
import pandas as pd
from torch.cuda.amp import autocast
from net import Net  # Assuming your model class is in a file named net.py
from torch.utils.data import DataLoader
from dataloader import DataLoaderPyTorch  # Assuming you have this custom dataloader

def load_model(model_path, device):
    model = Net().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict(model, test_loader, device):
    predictions = []
    
    model.eval()
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            
            with autocast():
                outputs = model(inputs)
                probs = torch.sigmoid(outputs.squeeze(1))
            
            preds = (probs > 0.5).float()
            
            predictions.extend(preds.cpu().numpy())
    
    return predictions

def create_submission_file(predictions, id_map_file, sample_submission_file, output_file):
    # Load ID mapping
    df_id_map = pd.read_csv(id_map_file)
    id_map = {row['id']: row['ID'] for _, row in df_id_map.iterrows()}
    
    # Load sample submission file
    sub = pd.read_csv(sample_submission_file)
    
    # Update predictions
    for i in range(len(sub)):
        sub.at[i, 'class'] = predictions[id_map[sub.at[i, 'id']]]
    
    # Save updated submission file
    sub.to_csv(output_file, index=False)
    print(f"Submission file saved to {output_file}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "last_checkpoint.pth"  # Adjust this path as needed
    test_file = "data/test_data.h5"
    batch_size = 32  # Adjust as needed
    id_map_file = 'id_map.csv'
    sample_submission_file = 'SampleSubmission.csv'
    output_file = 'submission.csv'
    
    # Load the model
    model = load_model(model_path, device)
    
    # Create test dataloader
    data_loader = DataLoaderPyTorch(train_file="data/train_data.h5", test_file=test_file, batch_size=batch_size)
    test_loader = data_loader.get_test_loader()
    test_dataset = test_loader.dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


    
    # Make predictions
    predictions = predict(model, test_loader, device)
    
    # Create submission file
    create_submission_file(predictions, id_map_file, sample_submission_file, output_file)

if __name__ == "__main__":
    main()