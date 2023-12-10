#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import os
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import matplotlib.pyplot as plt
from torchvision import transforms
import wandb
from predictor import Predictor
import random
import numpy as np
from lightning.pytorch.utilities import grad_norm
# from lightning.pytorch.tuner import Tuner

# In[3]:

class MaskPredDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split, ep_len=22, block_size=11, add_unlabeled=False):
        self.data_path = os.path.join(data_root, split)
        self.split = split
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
        ])
            
        self.block_size = block_size
        
        if split == "train":
            if add_unlabeled:
                self.masks = torch.cat([torch.load(os.path.join(data_root, "train_masks.pt")), 
                                              torch.load(os.path.join(data_root, "unlabeled_masks.pt")).squeeze()], dim=0)
            else:
                self.masks = torch.load(os.path.join(data_root, "train_masks.pt"))
        elif split == "val":
            self.masks = torch.load(os.path.join(data_root, "val_masks.pt"))
        
        self.seq_per_ep = ep_len - block_size

    def __len__(self):
        return len(self.masks) * self.seq_per_ep
                                    
    def get_video(self, idx):
        return self.masks[idx].long()
    
    def video_num(self):
        return len(self.masks)
                                    
    def __getitem__(self, idx):
        ep = idx // (self.seq_per_ep)
        offset = idx % (self.seq_per_ep)
        
        # read mask
        mask = self.masks[ep]
        mask = self.transform(mask)

        x = mask[offset:offset+self.block_size]
        y = mask[offset+1:offset+self.block_size+1]
        
        return x.long(), y.long()

class SampleVideoCallback(pl.pytorch.callbacks.Callback):
    def __init__(self, val_dataset):
        self.val_dataset = val_dataset

    def apply_cm(self, x):
        cm = plt.get_cmap()
        norm = plt.Normalize(vmin=x.min(), vmax=x.max())
        return cm(norm(x))[:, :, :3].transpose(2,0,1)

    def on_validation_epoch_end(self, trainer, pl_module):
        print("INFO: Sampling validation videos")
        # sample videos
        pl_module.eval()
        
        vid = self.val_dataset.get_video(random.randint(0, self.val_dataset.video_num()))
        x = vid[:11] # (T, H, W)
        x = x.unsqueeze(0).to(pl_module.device) # (1, T, H, W)
        
        for t in range(11):
            pred = pl_module(x)
            pred = torch.argmax(pred, dim=2)
            x = torch.cat([x[:,1:],pred[:, -1:]], dim=1)
        
        # forward pass
        x_hat = x.detach().cpu().squeeze(0)
        x_hat = torch.cat([vid[:11], x_hat], dim=0)

        # TODO: fix video generation here, put only autoregressive generation
        
        # # sample and log videos
        pred_imgs = []
        gt_imgs = []
        for t in range(x_hat.shape[0]):
            pred_img = self.apply_cm(x_hat[t])
            gt_img = self.apply_cm(vid[t])
            pred_imgs.append(pred_img)
            gt_imgs.append(gt_img)

        pred_imgs = np.stack(pred_imgs, axis=0)
        gt_imgs = np.stack(gt_imgs, axis=0)
        video = (np.concatenate([gt_imgs, pred_imgs], axis=-1) * 255).astype(np.uint8)
        trainer.logger.experiment.log({
            "val_video": wandb.Video(video, fps=4, format="gif")
        })

class MaskPredFormer(pl.LightningModule):
    def __init__(self, data_root, add_unlabeled, batch_size, lr, max_epochs):
        super().__init__()
        self.save_hyperparameters()
        self.predictor = Predictor()
        self.train_set = MaskPredDataset(data_root, "train", add_unlabeled=add_unlabeled)
        self.val_set = MaskPredDataset(data_root, "val")
    
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

    def forward(self, x):
        y_hat = self.predictor(x)
        return y_hat
    
    def step(self, batch):
        x, y = batch
        y_hat = self(x) # (b, t, 49, h, w)

        # compute loss
        b, t, *_ = y.shape
        y = y.view(b*t, *y.shape[2:]) # (b*t, h, w)
        y_hat = y_hat.view(b*t, *y_hat.shape[2:]) # (b*t, 49, h, w)

        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('train_loss', loss, rank_zero_only=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log('val_loss', loss, on_epoch=True, sync_dist=True)
        return loss
   
    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.predictor, norm_type=2)
        self.log_dict(norms, rank_zero_only=True)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 
#                                                          T_max=(len(self.train_set)//self.hparams.batch_size)*self.hparams.max_epochs, 
#                                                          eta_min=1e-5)
        return [optim] # , [sch]

# In[11]:
if __name__ == "__main__":
    pl.seed_everything(42)
    dirpath = "models_small_4gpu/"
    model = MaskPredFormer(data_root="/scratch/me2646/dataset", add_unlabeled=True, batch_size=8, lr=0.0005, max_epochs=10)

    logger = WandbLogger(project="mask-predformer")
    sample_video_cb = SampleVideoCallback(model.val_set)
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, 
                                        filename='small_model_4gpu_{epoch}-{val_loss:.3f}',
                                        monitor='val_loss', save_top_k=5, mode='min', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=model.hparams.max_epochs, accelerator="gpu", logger=logger,
                         log_every_n_steps=100,
                         val_check_interval=0.5,
                         gradient_clip_val=0.5,
                        callbacks=[sample_video_cb, checkpoint_callback, lr_monitor])
    
    ckpt_path = os.path.join(dirpath, "last.ckpt")
    trainer.fit(
        model,
        ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None
    )

