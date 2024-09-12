import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Capa convolucional 1
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)  # Normalización de batch
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling (reduce tamaño a la mitad)

        # Capa convolucional 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Capa convolucional 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Capa totalmente conectada (fully connected)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)  # 64 canales de características en una imagen de 2x2
        self.fc2 = nn.Linear(128, 1)  # Clasificación binaria

        # Dropout para regularización
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Cambiar de [batch_size, height, width, channels] a [batch_size, channels, height, width]
        x = x.permute(0, 3, 1, 2)

        # Aplicar la primera capa convolucional seguida de BatchNorm, ReLU y max pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Segunda capa convolucional
        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        # Tercera capa convolucional
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        # Aplanar para pasar a capas totalmente conectadas
        x = x.reshape(-1, 64 * 2 * 2)  # Usar reshape en lugar de view

        # Capa totalmente conectada con ReLU
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout para regularización

        # Capa final, salida binaria
        x = torch.sigmoid(self.fc2(x))  # Sigmoid para clasificar entre 0 y 1

        return x
