import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataloader import DataLoaderPyTorch  # Asegúrate de que este archivo contenga el DataLoader que has definido
from net import Net  # Importar el modelo que crearás en net.py
import torch.nn as nn

torch.backends.cudnn.benchmark = True

def train_model(model, train_loader, criterion, optimizer, device, max_batches_per_epoch=None):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
    batch_count = 0
    
    for inputs, labels in train_loader:
        if max_batches_per_epoch is not None and batch_count >= max_batches_per_epoch:
            break
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Resetear los gradientes
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs.squeeze(1), labels.float()) 

        loss.backward()  # Backward pass
        optimizer.step()  # Actualizar los parámetros
        
        # Actualizar estadísticas
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(batch_count)
        batch_count += 1
    
    train_loss = running_loss / total
    train_accuracy = correct / total
    return train_loss, train_accuracy

def validate_model(model, val_loader, criterion, device, max_batches_per_epoch=None):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0
    batch_count = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            if max_batches_per_epoch is not None and batch_count >= max_batches_per_epoch:
                break
            
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.squeeze(1), labels.float())  
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batch_count += 1
    
    val_loss = running_loss / total
    val_accuracy = correct / total
    return val_loss, val_accuracy

def main():
    # Parámetros
    epochs = 40  # Aumenta el número de épocas si lo deseas
    learning_rate = 0.001
    batch_size = 1
    validation_split = 0.2  # 20% de los datos para validación
    max_batches_per_epoch = 500  # Limitar el número de iteraciones por época
    
    # Preparar los datos
    train_file = "train_data.h5"
    test_file = "test_data.h5"
    
    # Crear DataLoader con balanceo de datos
    data_loader = DataLoaderPyTorch(train_file, test_file, batch_size=batch_size, balance_data=True)
    train_loader = data_loader.get_train_loader()
    
    # Dividir en conjunto de entrenamiento y validación
    total_train_samples = len(train_loader.dataset)
    val_size = int(total_train_samples * validation_split)
    train_size = total_train_samples - val_size
    
    train_data, val_data = random_split(train_loader.dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Cargar modelo
    model = Net().to(device)
    
    # Optimizador y función de pérdida
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Entrenamiento y validación
    for epoch in range(epochs):
        train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device, max_batches_per_epoch)
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device, max_batches_per_epoch)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
    
if __name__ == "__main__":
    main()
