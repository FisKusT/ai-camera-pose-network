import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import pytorch_lightning as pl

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class PoseDetector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.regression = nn.Linear(2048, 6)
        self.preprocess = weights.transforms()

    def forward(self, x):
        x = self.model(x)
        x = self.regression(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        output = self(x)
        loss = F.mse_loss(output, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.mse_loss(output, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer