# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/04_scheduled_sampling_trainer.ipynb.

# %% auto 0
__all__ = ['inv_sigmoid_schedule', 'MaskSimVPScheduledSamplingModule', 'SampleAutoRegressiveVideoCallback']

# %% ../nbs/04_scheduled_sampling_trainer.ipynb 2
import torch
import os

import lightning as pl
from torchmetrics import JaccardIndex

import wandb
import math
import random

from .mask_simvp import MaskSimVP
from .vis_utils import show_gif 
from .simvp_dataset import DLDataset 

# %% ../nbs/04_scheduled_sampling_trainer.ipynb 3
def inv_sigmoid_schedule(x, n, k):
    y = k / (k+math.exp(((x-(n//2))/(n//20))/k))
    return y

# %% ../nbs/04_scheduled_sampling_trainer.ipynb 4
class MaskSimVPScheduledSamplingModule(pl.LightningModule):
    def __init__(self, 
                 in_shape, hid_S, hid_T, N_S, N_T, model_type,
                 batch_size, lr, weight_decay, max_epochs, data_root,
                 sample_step_inc_every_n_epoch, schedule_k=2, max_sample_steps=5,
                 pre_seq_len=11, aft_seq_len=1, drop_path=0.0, unlabeled=False, downsample=False,
                ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MaskSimVP(
            in_shape, hid_S, hid_T, N_S, N_T, model_type, downsample=downsample, drop_path=drop_path,
            pre_seq_len=pre_seq_len, aft_seq_len=aft_seq_len
        )
        self.train_set = DLDataset(data_root, "train", unlabeled=unlabeled, pre_seq_len=pre_seq_len, aft_seq_len=max_sample_steps+1)
        self.val_set = DLDataset(data_root, "val", pre_seq_len=11, aft_seq_len=11)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.iou_metric = JaccardIndex(task='multiclass', num_classes=49)
        
        self.schedule_idx = 0
        self.sample_steps = 1
        self.schedule_max = len(self.train_dataloader()) * sample_step_inc_every_n_epoch

    def sample_or_not(self):
        assert self.schedule_idx < self.schedule_max, "Schedule idx larger than max, something wrong with schedule"

        p = 1 - inv_sigmoid_schedule(self.schedule_idx, self.schedule_max, self.hparams.schedule_k)
        self.schedule_idx += 1
        return random.random() < p
    
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

    @torch.no_grad()
    def sample_autoregressive(self, x, t):
        cur_seq = x.clone()
        for _ in range(t):
            y_hat_logit_t = self.model(cur_seq)
            y_hat = torch.argmax(y_hat_logit_t, dim=2) # get current prediction
            cur_seq = torch.cat([cur_seq[:, 1:], y_hat], dim=1) # autoregressive concatenation
        return cur_seq
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.sample_or_not():
            x = self.sample_autoregressive(x, self.sample_steps)
            y = y[:, self.sample_steps:self.sample_steps+1] # get the next label after sampling model `sample_steps` times
        else:
            # no change in x
            y = y[:, 0:1] # get the normal training label
        
        y_hat_logits = self.model(x)
        
        # Flatten batch and time dimensions
        b, t, *_ = y_hat_logits.shape
        y_hat_logits = y_hat_logits.view(b*t, *y_hat_logits.shape[2:])
        y = y.view(b*t, *y.shape[2:])

        loss = self.criterion(y_hat_logits, y)
        
        self.log("train_loss", loss)
        self.log("sample_steps", self.sample_steps)
        self.log("schedule_idx", self.schedule_idx)
        self.log("schedule_prob", 1 - inv_sigmoid_schedule(self.schedule_idx, self.schedule_max, self.hparams.schedule_k))
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.sample_autoregressive(x, 11)
        iou = self.iou_metric(y_hat[:, -1], y[:, -1])
        self.log("valid_last_frame_iou", self.iou_metric, on_step=False, on_epoch=True, sync_dist=True)
        return iou

    def on_train_epoch_end(self):
        if (self.current_epoch+1) % self.hparams.sample_step_inc_every_n_epoch:
            print("Increasing sample steps")
            self.schedule_idx = 0
            self.sample_steps = max(self.sample_steps+1, self.hparams.max_sample_steps)
        
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

# %% ../nbs/04_scheduled_sampling_trainer.ipynb 13
class SampleAutoRegressiveVideoCallback(pl.Callback):
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
        
        cur_seq = pl_module.sample_autoregressive(x, 11)

        # convert to numpy
        x = x.squeeze(0).cpu().numpy()
        y = y.squeeze(0).cpu().numpy()
        y_hat = cur_seq.squeeze(0).cpu().numpy()

        gif_path = os.path.join(self.val_path, f"val_ar_video_{self.val_count}.gif")

        show_gif(x, y, y_hat, out_path=gif_path)
        self.val_count += 1

        return gif_path
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.global_rank == 0:
            gif_path = self.generate_video(pl_module)
            trainer.logger.experiment.log({
                "val_video": wandb.Video(gif_path, fps=4, format="gif")
            })
