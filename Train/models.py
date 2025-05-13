import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder como ModuleList
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 16),    # encode1
            DoubleConv(16, 32),             # encode2
            DoubleConv(32, 64),             # encode3
            DoubleConv(64, 128)             # encode4
        ])
        self.pooldown = nn.ModuleList([
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        ])

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder como ModuleList
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])
        self.decoder = nn.ModuleList([
            DoubleConv(256, 128),  # concat de c4 + u6
            DoubleConv(128, 64),   # concat de c3 + u7
            DoubleConv(64, 32),    # concat de c2 + u8
            DoubleConv(32, 16)     # concat de c1 + u9
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
    def __init__(self, in_channels=3, out_channels=3):
        super(Auto, self).__init__()

        # Encoder como ModuleList
        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 16),    # encode1
            DoubleConv(16, 32),             # encode2
            DoubleConv(32, 64),             # encode3
            DoubleConv(64, 128)             # encode4
        ])
        self.pooldown = nn.ModuleList([
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        ])

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Decoder como ModuleList
        self.upconv = nn.ModuleList([
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        ])
        self.decoder = nn.ModuleList([
            DoubleConv(128, 128),  
            DoubleConv(64, 64),   
            DoubleConv(32, 32),    
            DoubleConv(16, 16)     
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