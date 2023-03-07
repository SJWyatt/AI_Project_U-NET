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
        inputs = batch["ct_scan"].unsqueeze(1)
        masks = batch["masks"]

        outs = self(inputs)
        
        loss_ce = F.cross_entropy(outs, masks)

        # TODO: Also use dice loss to improve segmentation.
        # loss_dice = dice_loss(
        #     F.softmax(outs, dim=1).float(), 
        #     masks.permute(0, 3, 1, 2).float()
        # )

        loss = loss_ce# + loss_dice
        self.log('train_loss', loss, batch_size=self.batch_size)
        # self.log('train_ce_loss', loss_ce, batch_size=self.batch_size)
        # self.log('train_dice_loss', loss_dice, batch_size=self.batch_size)

        return loss
    
    def training_step_end(self, step_output):
        return super().training_step_end(step_output)

    def validation_step(self, batch, batch_idx):
        inputs:Tensor = batch["ct_scan"].unsqueeze(1)
        masks:Tensor = batch["masks"]
        
        outs = self(inputs)

        loss_ce = F.cross_entropy(outs, masks)

        # TODO: Also use dice loss to improve segmentation.
        # loss_dice = dice_loss(
        #     F.softmax(outs, dim=1).float(), 
        #     masks.permute(0, 3, 1, 2).float()
        # )

        loss = loss_ce # + loss_dice
        # self.log('val_loss', loss, batch_size=self.batch_size)
        # self.log('val_ce_loss', loss_ce, batch_size=self.batch_size)
        # self.log('val_dice_loss', loss_dice, batch_size=self.batch_size)

        return loss

    def validation_epoch_end(self, epoch_output):
        loss = torch.stack(epoch_output).mean()
        self.log('avg_val_loss', loss, batch_size=self.batch_size)
        
        return super().validation_epoch_end(epoch_output)



if __name__ == "__main__":
    cli = LightningCLI(UNetTrainer, IrcadDataloader, save_config_callback=None)
