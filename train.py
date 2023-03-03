import torch
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

from loss import dice_loss
from model import UNet
from dataset import IrcadDataloader


class UNetTrainer(pl.LightningModule):
    def __init__(self,
        # Training Parameters
        batch_size:int=32,

        # Optimizer Parameters
        learning_rate:float=1e-6,
        
        # Model Parameters
        num_channels:int=1,
        num_classes:int=10,
        bilinear:bool=False,

        # Logging Parameters
        verbose:bool=False
    ):
        super().__init__()
        self.verbose = verbose

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        self.model = UNet(n_channels=num_channels, n_classes=self.num_classes, bilinear=bilinear)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        return optimizer

    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs = batch["ct_scan"]
        masks = batch["masks"]

        # Convert separate masks into a single mask with each class as a channel.
        # masks = torch.stack([masks == i for i in range(self.num_classes)], dim=1).float()
        masks = torch.stack(masks, dim=1).float()

        outs = self(inputs)
        
        loss = F.cross_entropy(outs, masks)
        # Also use dice loss for better segmentation.
        loss += dice_loss(outs, masks)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["ct_scan"]
        masks = batch["masks"]

        # Convert separate masks into a single mask with each class as a channel.
        masks = torch.stack(masks, dim=1).float()

        outs = self(inputs)

        loss = F.cross_entropy(outs, masks)
        # Also use dice loss for better segmentation.
        loss += dice_loss(outs, inputs)

        self.log('val_loss', loss)

        return loss

if __name__ == "__main__":
    cli = LightningCLI(UNetTrainer, IrcadDataloader)

