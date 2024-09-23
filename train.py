import torch
import torch.nn as nn
import torch.optim.lr_scheduler as StepLR

import torch.optim as optim
from torch.utils.data import Subset, random_split, DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import random
import os
import wandb
import timm
from torchgeo.models import ResNet18_Weights

from dataloader import DataLoaderPyTorch
from net import Net

torch.backends.cudnn.benchmark = True

def train_model(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        predicted = ((torch.sigmoid(outputs.squeeze(1))) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / total, correct / total

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(1), labels.float())
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return running_loss / total, correct / total

def save_checkpoint(epoch, model, optimizer, scaler, loss, accuracy, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def load_checkpoint(filename, model, optimizer, scaler):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        print(f"Checkpoint loaded: {filename}")
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch, loss, accuracy
    else:
        print(f"No checkpoint found at: {filename}")
        return 0, float('inf'), 0

def modify_resnet_gradual(model, num_input_channels=6, freeze_layers=True):
    

    # Modificamos la primera capa conv para 6 canales
    model.conv1 = nn.Conv2d(num_input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Eliminar max pooling
    #model.fc = nn.Linear(model.fc.in_features, 1)  # Ajustar la salida
    
    # Congelamos todas las capas inicialmente
    if freeze_layers:
        for param in model.parameters():
            param.requires_grad_(False)

    # Descongelamos ciertas capas clave progresivamente
    model.layer3.requires_grad_(True)  # Descongelamos la capa 3 
    model.fc.requires_grad_(True)      # Descongelamos la capa final
    
    # Mantener conv1 también ajustable si es necesario
    model.conv1.requires_grad_(True)
    
    return model


def main():
    wandb.init(project="indios")

    # Parameters
    epochs = 10000
    learning_rate = 1e-6
    batch_size = 512
    num_workers = 8
    samples_per_epoch = 2048
    checkpoint_dir = "checkpoints"
    
    wandb.config.update({
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch
    })

    train_file = "data/train_split.h5"
    validation_file = "data/validation_split.h5"
    
    data_loader_train = DataLoaderPyTorch(
        train_file,
        batch_size=batch_size, 
        num_workers=num_workers,
        subset=1.0,
        target_ratio=0.6,
        augment=True,  # Solo para entrenamiento
        balance_data=True
    )

    # DataLoader para validación/test sin augmentación
    data_loader_val = DataLoaderPyTorch(
        validation_file,
        batch_size=batch_size, 
        num_workers=num_workers,
        subset=1.0,
        target_ratio=0.6,
        augment=False,  
        balance_data=True 
    )

    full_train_loader = data_loader_train.get_train_loader()
    full_val_loader = data_loader_val.get_train_loader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = Net()
    weights = ResNet18_Weights.LANDSAT_ETM_SR_MOCO
    model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"], num_classes=10)
    #print(model)
    model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    model = modify_resnet_gradual(model, num_input_channels=6)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    scaler = GradScaler()
    
    wandb.watch(model)
    init_epoch=10
    print("Phase 1: Training only final layer")
    for epoch in range(init_epoch):
        train_subset_loader = DataLoader(
            Subset(full_train_loader.dataset, random.sample(range(len(full_train_loader.dataset)), samples_per_epoch * 4)),
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
        val_subset_loader = DataLoader(
            Subset(full_val_loader.dataset, random.sample(range(len(full_val_loader.dataset)), samples_per_epoch)),
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )
        
        train_loss, train_accuracy = train_model(model, train_subset_loader, criterion, optimizer, device, scaler) 
        val_loss, val_accuracy = validate_model(model, val_subset_loader, criterion, device)
       
        wandb.log({
            "epoch": epoch + 1,
            "phase": 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        print(f'Epoch [{epoch+1}/{init_epoch}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
    print("Phase 2:  Fine-tune all layers")
    for param in model.parameters():
        param.requires_grad_(True)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)


    best_loss = float('inf')
    for epoch in range(init_epoch, epochs):
        train_subset_loader = DataLoader(
            Subset(full_train_loader.dataset, random.sample(range(len(full_train_loader.dataset)), samples_per_epoch*4)),
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
        val_subset_loader = DataLoader(
            Subset(full_val_loader.dataset, random.sample(range(len(full_val_loader.dataset)), samples_per_epoch)),
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True
        )
        
        train_loss, train_accuracy = train_model(model, train_subset_loader, criterion, optimizer, device, scaler)
        val_loss, val_accuracy = validate_model(model, val_subset_loader, criterion, device)
        scheduler.step()

        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        save_checkpoint(epoch, model, optimizer, scaler, val_loss, val_accuracy, os.path.join(checkpoint_dir, "last_checkpoint.pth"))
        
        wandb.log({
            "epoch": epoch + 1,
            "phase": 2,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(epoch, model, optimizer, scaler, val_loss, val_accuracy, os.path.join(checkpoint_dir, "best_model.pth"))

    wandb.finish()

if __name__ == "__main__":
    main()