import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.0, kernel_size = 3):
        super(DoubleConv, self).__init__()
        padding = kernel_size//2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        ]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout2d(dropout_rate))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)

class CNN(nn.Module):
    def __init__(self, dropout_rate=0.0, in_channels=1, out_classes=9, img_heigth=28, img_width=28):
        super(CNN, self).__init__()
        self.img_heigth = (img_heigth//16)*16
        self.img_width = (img_width//16)*16
        
        # Encoder
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 16, dropout_rate),
            DoubleConv(16, 32, dropout_rate),
            DoubleConv(32, 64, dropout_rate),
            DoubleConv(64, 128, dropout_rate)
        ])
        self.pooldown = nn.ModuleList([
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        ])

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256, dropout_rate)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(self.img_heigth*self.img_width, out_classes*out_classes),
            nn.ReLU(inplace=True),
            nn.Linear(out_classes*out_classes, out_classes)
            )

    def forward(self, x):
        # Encoder
        for i in range(4):
            x = self.encoder[i](x)
            x = self.pooldown[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Classification head
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class NetCNN(nn.Module):

    def __init__(self, dropout_rate = 0.0, out_classes=9, img_heigth=28, img_width=28):
        self.img_heigth = img_heigth//4-3
        self.img_width = img_width//4-3
        super(NetCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*self.img_heigth*self.img_width, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # Al pasar de capa convolucional a capa totalmente conectada, tenemos
        # que reformatear la salida para que se transforme en un vector unidimensional
        x = x.view(-1, 16*self.img_heigth*self.img_width)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NetMLP(torch.nn.Module):
    def __init__(self, dropout_rate = 0.0, out_classes=9, img_heigth=28, img_width=28, hidden_layer = 15):
        super(NetMLP, self).__init__()
        self.hidden1 = nn.Linear(img_heigth*img_width, hidden_layer)
        self.hidden2 = nn.Linear(hidden_layer, hidden_layer)
        self.out = nn.Linear(hidden_layer, out_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x