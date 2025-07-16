import torch
from torch import nn

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

class UNet(nn.Module):
    def __init__(self, dropout_rate=0.0, in_channels=3, out_channels=1, out_classes=9, img_heigth=28, img_width=28):
        super(UNet, self).__init__()

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

        # Decoder
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])
        self.decoder = nn.ModuleList([
            DoubleConv(256, 128, dropout_rate),  # concat de c4 + u6
            DoubleConv(128, 64, dropout_rate),   # concat de c3 + u7
            DoubleConv(64, 32, dropout_rate),    # concat de c2 + u8
            DoubleConv(32, 16, dropout_rate)     # concat de c1 + u9
        ])

        # Final layer
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        for i in range(4):
            x = self.encoder[i](x)
            skips.append(x)
            x = self.pooldown[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(4):
            x = self.upconv[i](x)
            skip = skips[-(i+1)]
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i](x)

        # Final layer
        return self.final(x)

class Auto(nn.Module):
    def __init__(self, dropout_rate=0.0, in_channels=3, out_channels=3, out_classes=9, img_heigth=28, img_width=28):
        super(Auto, self).__init__()

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

        # Decoder
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])
        self.decoder = nn.ModuleList([
            DoubleConv(128, 128, dropout_rate),  
            DoubleConv(64, 64, dropout_rate),   
            DoubleConv(32, 32, dropout_rate),    
            DoubleConv(16, 16, dropout_rate)     
        ])

        # Final layer
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        for i in range(4):
            x = self.encoder[i](x)
            x = self.pooldown[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(4):
            x = self.upconv[i](x)
            x = self.decoder[i](x)

        # Final layer
        return self.final(x)

class ShortUNet(nn.Module):
    def __init__(self, dropout_rate=0.0, in_channels=3, out_channels=1, out_classes=9, img_heigth=28, img_width=28):
        super(ShortUNet, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 16, dropout_rate),
            DoubleConv(16, 32, dropout_rate),
            DoubleConv(32, 64, dropout_rate)
        ])
        self.pooldown = nn.ModuleList([
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        ])

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128, dropout_rate)

        # Decoder
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])
        self.decoder = nn.ModuleList([
            DoubleConv(128, 64, dropout_rate),   # concat de c3 + u7
            DoubleConv(64, 32, dropout_rate),    # concat de c2 + u8
            DoubleConv(32, 16, dropout_rate)     # concat de c1 + u9
        ])

        # Final layer
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skips = []
        for i in range(3):
            x = self.encoder[i](x)
            skips.append(x)
            x = self.pooldown[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(3):
            x = self.upconv[i](x)
            skip = skips[-(i+1)]
            x = torch.cat([skip, x], dim=1)
            x = self.decoder[i](x)

        # Final layer
        return self.final(x)

class ShortAuto(nn.Module):
    def __init__(self, dropout_rate=0.0, in_channels=3, out_channels=3, out_classes=9, img_heigth=28, img_width=28):
        super(ShortAuto, self).__init__()

        # Encoder
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 16, dropout_rate),
            DoubleConv(16, 32, dropout_rate),
            DoubleConv(32, 64, dropout_rate)
        ])
        self.pooldown = nn.ModuleList([
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        ])

        # Bottleneck
        self.bottleneck = DoubleConv(64, 128, dropout_rate)

        # Decoder
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])
        self.decoder = nn.ModuleList([ 
            DoubleConv(64, 64, dropout_rate),   
            DoubleConv(32, 32, dropout_rate),    
            DoubleConv(16, 16, dropout_rate)     
        ])

        # Final layer
        self.final = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        for i in range(3):
            x = self.encoder[i](x)
            x = self.pooldown[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(3):
            x = self.upconv[i](x)
            x = self.decoder[i](x)

        # Final layer
        return self.final(x)

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
        super(NetCNN, self).__init__()
        self.img_heigth = img_heigth//4-3
        self.img_width = img_width//4-3

        # Encoder
        self.encoder = nn.ModuleList([
            nn.Conv2d(1, 6, 5),
            nn.Conv2d(6, 16, 5)
        ])
        self.pooldown = nn.ModuleList([
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        ])

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(16*self.img_heigth*self.img_width, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, out_classes),
            )

    def forward(self, x):
        # Encoder
        for i in range(2):
            x = self.encoder[i](x)
            x = self.pooldown[i](x)

        # Classification head
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class NetMLP(torch.nn.Module):
    def __init__(self, dropout_rate = 0.0, out_classes=9, img_heigth=28, img_width=28, hidden_layer = 15):
        super(NetMLP, self).__init__()
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(img_heigth*img_width, hidden_layer),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer, out_classes),
            )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x