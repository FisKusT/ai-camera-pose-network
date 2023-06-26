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

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        # self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(2048, 6)
        # self.fc2 = nn.Linear(512, 6)
        # self.relu = nn.ReLU()

    def forward(self, x):
        # x = self.dropout(x)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        return x

class PoseDetector(pl.LightningModule):
    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.model.fc = Identity()
        self.regression = Regression()
        # self.preprocess = weights.transforms()

    def forward(self, x):
        x = self.model(x)
        x = self.regression(x)
        return x

    # def compute_loss(self, output, target):
    #     loss = .mse_loss(output, target)
    #     return loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        output = self(x)
        loss = F.mse_loss(output, y)
        self.log('train_loss', loss, sync_dist=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.mse_loss(output, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer