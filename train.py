import torch
import torch.nn.functional as F
from torch import Tensor
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
import wandb

from loss import dice_loss
from model import UNet
from dataset import IrcadDataloader


def calculate_iou(pred_masks:Tensor, gt_masks:Tensor, smooth=1):
    # Helpful article: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

    # Calculate the intersection over union for the mask and the prediction
    # pred_masks = (F.softmax(pred_masks, dim=1) > torch.tensor(0.5)).float()
    # intersection = (pred_masks & gt_masks).sum()
    # union = (pred_masks | gt_masks).sum()

    # return intersection / union

    intersection = torch.sum(torch.abs(gt_masks * pred_masks), dim=[1,2])
    union = torch.sum(gt_masks, [1,2]) + torch.sum(pred_masks, [1,2]) - intersection
    return torch.mean((intersection + smooth) / (union + smooth), dim=0)

def calculate_dice(pred_masks:Tensor, gt_masks:Tensor, smooth=1):
    intersection = torch.sum(torch.abs(gt_masks * pred_masks), dim=[1,2])
    union = torch.sum(gt_masks, [1,2]) + torch.sum(pred_masks, [1,2])

    return torch.mean((2. * intersection + smooth) / (union + smooth), dim=0)

def calculate_pixel_accuracy(pred_masks:Tensor, gt_masks:Tensor):
    return torch.sum((pred_masks.round() == gt_masks), [1, 2]).float() / (gt_masks.shape[1] * gt_masks.shape[2])

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
        inputs:Tensor = batch["ct_scan"].unsqueeze(1)
        masks:Tensor = batch["masks"]

        outs:Tensor = self(inputs)
        
        if self.num_classes > 1:
            loss_ce = F.cross_entropy(outs, masks)
            self.log('train_ce_loss', loss_ce, batch_size=self.batch_size, sync_dist=True)

            loss_dice = dice_loss(
                F.softmax(outs, dim=1).float(), 
                F.one_hot(masks, self.num_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            self.log('train_dice_loss', loss_dice, batch_size=self.batch_size, sync_dist=True)

            loss = loss_ce + loss_dice
        else:
            outs = outs.squeeze(1)
            out_sigmoid = torch.sigmoid(outs)

            loss_bce = F.binary_cross_entropy_with_logits(outs, masks)
            self.log('train_bce_loss', loss_bce, batch_size=self.batch_size, sync_dist=True)

            loss_dice = dice_loss(
                out_sigmoid,
                masks.float(),
                multiclass=False,
            )
            self.log('train_dice_loss', loss_dice, batch_size=self.batch_size, sync_dist=True)

            loss = loss_bce + loss_dice #+ loss_ce

            iou = calculate_iou(out_sigmoid, masks.float())
            self.log('train_iou', iou, batch_size=self.batch_size, sync_dist=True)

            dice = calculate_dice(out_sigmoid, masks.float())
            self.log('train_dice', dice, batch_size=self.batch_size, sync_dist=True)

            # Report pixel accuracy
            acc = calculate_pixel_accuracy(out_sigmoid, masks.float()).sum() / self.batch_size
            self.log('train_acc', acc, batch_size=self.batch_size, sync_dist=True)

        # Log the aggregated loss
        self.log('train_loss', loss, batch_size=self.batch_size, sync_dist=True)

        return loss
    
    def training_step_end(self, step_output):
        return super().training_step_end(step_output)

    def validation_step(self, batch, batch_idx):
        inputs:Tensor = batch["ct_scan"].unsqueeze(1)
        masks:Tensor = batch["masks"]
        
        outs:Tensor = self(inputs)

        ret_data = {}
        if self.num_classes > 1:
            loss_ce = F.cross_entropy(outs, masks)
            self.log('val_ce_loss', loss_ce, batch_size=self.batch_size, sync_dist=True)

            loss_dice = dice_loss(
                F.softmax(outs, dim=1).float(), 
                F.one_hot(masks, self.num_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )
            self.log('val_dice_loss', loss_dice, batch_size=self.batch_size, sync_dist=True)

            ret_data['loss'] = loss_ce + loss_dice
        else:
            outs = outs.squeeze(1)
            out_sigmoid = torch.sigmoid(outs)

            loss_bce = F.binary_cross_entropy_with_logits(outs, masks)
            self.log('val_bce_loss', loss_bce, batch_size=self.batch_size, sync_dist=True)

            loss_dice = dice_loss(
                out_sigmoid, 
                masks.float(), 
                multiclass=False
            )
            self.log('val_dice_loss', loss_dice, batch_size=self.batch_size, sync_dist=True)

            ret_data['loss'] = loss_bce + loss_dice
            ret_data['iou'] = calculate_iou(out_sigmoid, masks)
            ret_data['dice'] = calculate_dice(out_sigmoid, masks)
            ret_data['acc'] = calculate_pixel_accuracy(out_sigmoid, masks).mean()

        if batch_idx == 0:
            images = []
            for img, pred, true in zip(inputs, outs, masks):
                pred = (torch.sigmoid(pred) > torch.tensor(0.5).to(self.device)).to(torch.uint8) 
                true = true.to(torch.uint8)

                images.append(wandb.Image(img.detach().cpu().numpy(), 
                    masks={
                        "predictions": {
                            "mask_data": pred.detach().cpu().numpy(),
                            "class_labels": {
                                0: "background",
                                1: "lungs",
                            }
                        },
                        "ground_truth": {
                            "mask_data": true.detach().cpu().numpy(),
                            "class_labels": {
                                0: "background",
                                1: "lungs",
                            }
                        },
                    })
                )
            self.logger.experiment.log({"examples": images, "epoch": self.current_epoch, "batch": batch_idx})
            # self.log("examples", images, batch_size=self.batch_size, sync_dist=True)

        # Log the aggregated loss in the epoch_end function
        # self.log('val_loss', loss, batch_size=self.batch_size, sync_dist=True)

        return ret_data

    def validation_epoch_end(self, epoch_output):
        # Log the average loss over the validation set
        loss = torch.stack([out['loss'] for out in epoch_output]).mean()
        self.log('avg_val_loss', loss, batch_size=self.batch_size, sync_dist=True)

        if self.num_classes == 1:
            iou = torch.stack([out['iou'] for out in epoch_output]).mean()
            self.log('avg_val_iou', iou, batch_size=self.batch_size, sync_dist=True)

            dice = torch.stack([out['dice'] for out in epoch_output]).mean()
            self.log('avg_val_dice', dice, batch_size=self.batch_size, sync_dist=True)

            acc = torch.stack([out['acc'] for out in epoch_output]).mean()
            self.log('avg_val_acc', acc, batch_size=self.batch_size, sync_dist=True)
        
        return super().validation_epoch_end(epoch_output)


if __name__ == "__main__":
    cli = LightningCLI(UNetTrainer, IrcadDataloader, save_config_callback=None, subclass_mode_data=True)
