from torch import nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class Unet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.n_classes = 1
        self.sigmoid=nn.Sigmoid()
        
        self.model = smp.UnetPlusPlus('timm-regnety_120', classes=self.n_classes)
        
    def forward(self, batch):
        x = self.model(batch)
        return self.sigmoid(x)