import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, random_split, DataLoader
from dataloader import DataLoaderPyTorch
from net import Net 
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import wandb



torch.backends.cudnn.benchmark = True

def train_model(model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Entrenando"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(1), labels.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * inputs.size(0)
        predicted = (torch.sigmoid(outputs.squeeze(1)) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return running_loss / total, correct / total

def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validando"):
            if isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    inputs, labels = batch
                else:
                    print(f"Advertencia: El lote tiene {len(batch)} elementos. Se esperaban 2.")
                    inputs, labels = batch[:2]  # Tomar solo los dos primeros elementos
            else:
                print(f"Advertencia: El lote no es una tupla o lista. Tipo: {type(batch)}")
                continue  # Saltar este lote

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
    print(f"Checkpoint guardado: {filename}")

def load_checkpoint(filename, model, optimizer, scaler):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        print(f"Checkpoint cargado: {filename}")
        print(f"Reanudando desde la época {start_epoch}")
        return start_epoch, loss, accuracy
    else:
        print(f"No se encontró el checkpoint: {filename}")
        return 0, 0, 0

def main():

    wandb.init(project="indios")

    # Parámetros
    epochs = 4000
    learning_rate = 0.001
    batch_size = 128
    num_workers = 4
    samples_per_epoch = 20  # Número de muestras a usar por época
    checkpoint_dir = "checkpoints"
    
    # Log hyperparameters
    wandb.config.update({
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch
    })

    # Crear directorio para checkpoints si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Preparar los datos
    train_file = "data/train_data.h5"
    test_file = "data/test_data.h5"
    
    data_loader = DataLoaderPyTorch(train_file, test_file, batch_size=batch_size, num_workers=num_workers)
    full_train_loader = data_loader.get_train_loader()
    
    # Dividir el conjunto de entrenamiento en entrenamiento y validación
    full_dataset = full_train_loader.dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Usar el nuevo modelo GradientBoostedCNNRF
    #model = MetaModel().to(device)    #model = MetaModel().to(device)
    model = Net().to(device)

    wandb.watch(model)  # Watch the model to log gradients and model parameters

    print(f"Información del modelo: {model}")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    
    # Intentar cargar el último checkpoint
    last_checkpoint = os.path.join(checkpoint_dir, "last_checkpoint.pth")
    start_epoch, best_loss, best_accuracy = load_checkpoint(last_checkpoint, model, optimizer, scaler)
    """# Función para guardar los 6 canales de la imagen en 6 imágenes separadas
    def guardar_canales(imagen, nombre_base):
        for i in range(6):
            canal = imagen[i].cpu().numpy()
            plt.imsave(f"{nombre_base}_canal_{i}.png", canal, cmap='gray')

    # Guardar los canales de una imagen de muestra
    muestra_batch, _ = next(iter(train_loader))
    muestra_imagen = muestra_batch[0]  # Tomamos la primera imagen del lote
    guardar_canales(muestra_imagen, "muestra")"""

    print("Se han guardado los 6 canales de una imagen de muestra.")
    for epoch in range(start_epoch, epochs):
        # Crear un nuevo subconjunto aleatorio para cada época
        train_subset_indices = random.sample(range(len(train_dataset)), samples_per_epoch*4)
        train_subset = Subset(train_dataset, train_subset_indices)
        train_subset_loader = DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True
        )
        
        # Crear un nuevo subconjunto aleatorio para validación, similar al de entrenamiento
        val_subset_indices = random.sample(range(len(val_dataset)), samples_per_epoch)
        val_subset = Subset(val_dataset, val_subset_indices)
        val_subset_loader = DataLoader(
            val_subset, 
            batch_size=batch_size, 
            shuffle=False,  # No es necesario mezclar el conjunto de validación
            num_workers=num_workers, 
            pin_memory=True
        )
        
        train_loss, train_accuracy = train_model(model, train_subset_loader, criterion, optimizer, device, scaler)
        
        
        val_loss, val_accuracy = validate_model(model, val_subset_loader, criterion, device)
        
        print(f'Época [{epoch+1}/{epochs}]')
        print(f'Pérdida de entrenamiento: {train_loss:.4f}, Precisión de entrenamiento: {train_accuracy:.4f}')
        print(f'Pérdida de validación: {val_loss:.4f}, Precisión de validación: {val_accuracy:.4f}')
        
        # Guardar checkpoint después de cada época
        #save_checkpoint(epoch, model, optimizer, scaler, val_loss, val_accuracy, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
        
        # Guardar el último checkpoint (sobrescribir)
        save_checkpoint(epoch, model, optimizer, scaler, val_loss, val_accuracy, last_checkpoint)
        
        # Opcionalmente, guardar el mejor modelo basado en la pérdida de validación
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(epoch, model, optimizer, scaler, val_loss, val_accuracy, os.path.join(checkpoint_dir, "best_model.pth"))
    # Finish the wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
