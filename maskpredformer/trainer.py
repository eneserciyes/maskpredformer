# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/02_trainer.ipynb.

# %% auto 0
__all__ = ['MaskSimVPModule', 'SampleVideoCallback']

# %% ../nbs/02_trainer.ipynb 2
import torch
import os

import lightning as pl

import matplotlib.pyplot as plt
import wandb
import random

from .mask_simvp import MaskSimVP, DEFAULT_MODEL_CONFIG
from .simvp_dataset import DLDataset
from .vis_utils import show_gif
from .simvp_dataset import DLDataset, DEFAULT_DATA_PATH

# %% ../nbs/02_trainer.ipynb 3
class MaskSimVPModule(pl.LightningModule):
    def __init__(self, 
                 in_shape, hid_S, hid_T, N_S, N_T, model_type,
                 batch_size, lr, weight_decay, max_epochs,
                 data_root, unlabeled=False, downsample=False):
        super().__init__()
        self.save_hyperparameters()
        self.model = MaskSimVP(
            in_shape, hid_S, hid_T, N_S, N_T, model_type, downsample=downsample
        )
        self.train_set = DLDataset(data_root, "train", unlabeled=unlabeled)
        self.val_set = DLDataset(data_root, "val")
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, 
            num_workers=8, shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_set, batch_size=self.hparams.batch_size, 
            num_workers=8, shuffle=False, pin_memory=True
        )

    def step(self, x, y):
        y_hat_logits = self.model(x)
        return y_hat_logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.step(x, y)
        
        # Flatten batch and time dimensions
        b, t, *_ = y_hat_logits.shape
        y_hat_logits = y_hat_logits.view(b*t, *y_hat_logits.shape[2:])
        y = y.view(b*t, *y.shape[2:])

        loss = self.criterion(y_hat_logits, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat_logits = self.step(x, y)

        # Flatten batch and time dimensions
        b, t, *_ = y_hat_logits.shape
        y_hat_logits = y_hat_logits.view(b*t, *y_hat_logits.shape[2:])
        y = y.view(b*t, *y.shape[2:])
       
        loss = self.criterion(y_hat_logits, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.hparams.lr,
            total_steps=self.hparams.max_epochs*len(self.train_dataloader()),
            final_div_factor=1e4
        )
        opt_dict = {
            "optimizer": optimizer,
            "lr_scheduler":{
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            } 
        }

        return opt_dict

# %% ../nbs/02_trainer.ipynb 9
class SampleVideoCallback(pl.Callback):
    def __init__(self, val_set, video_path="./val_videos/"):
        super().__init__()
        self.val_set = val_set
        self.val_count = 0
        self.val_path = video_path
        if not os.path.exists(self.val_path):
            os.makedirs(self.val_path)

    def generate_video(self, pl_module):
        pl_module.eval()
        sample_idx = random.randint(0, len(self.val_set)-1)
        
        x, y = self.val_set[sample_idx]
        x = x.unsqueeze(0).to(pl_module.device)
        y = y.unsqueeze(0).to(pl_module.device)

        y_hat_logits = pl_module.step(x,y).squeeze(0) # (T, 49, H, W)
        y_hat = torch.argmax(y_hat_logits, dim=1) # (T, H, W)

        # convert to numpy
        x = x.squeeze(0).cpu().numpy()
        y = y.squeeze(0).cpu().numpy()
        y_hat = y_hat.cpu().numpy()

        gif_path = os.path.join(self.val_path, f"val_video_{self.val_count}.gif")

        show_gif(x, y, y_hat, out_path=gif_path)
        self.val_count += 1

        return gif_path
    
    def on_validation_epoch_end(self, trainer, pl_module):
        gif_path = self.generate_video(pl_module)
        trainer.logger.experiment.log({
            "val_video": wandb.Video(gif_path, fps=4, format="gif")
        })


