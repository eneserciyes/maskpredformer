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
# from lightning.pytorch.tuner import Tuner


# In[3]:

class MaskPredDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, split, ep_len=22, block_size=11):
        self.data_path = os.path.join(data_root, split)
        self.split = split
        self.transform = transforms.Resize((80,120), interpolation=transforms.InterpolationMode.NEAREST, antialias=True)
        self.block_size = block_size

        all_videos = [os.path.join(self.data_path, name) for name in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, name))]
        all_videos.sort(key= lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

        if split == "train":
            all_videos_unlabeled = [
                os.path.join(data_root, "unlabeled", name) for name in 
                os.listdir(os.path.join(data_root, "unlabeled")) if os.path.isdir(os.path.join(data_root, "unlabeled", name))
            ]
            all_videos_unlabeled.sort(key= lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
            all_videos += all_videos_unlabeled

        self.all_masks = [os.path.join(video, "mask.pt") for video in all_videos]
        self.all_masks = torch.stack([torch.load(mask_path) for mask_path in self.all_masks], dim=0)
        self.seq_per_ep = ep_len - block_size

    def __len__(self):
        return len(self.all_videos) * self.seq_per_ep

    def __getitem__(self, idx):
        ep = idx // (self.block_size+1)
        offset = idx % (self.block_size+1)
        
        # read mask
        mask = self.all_masks[ep]
        mask = self.transform(mask)

        x = mask[offset:offset+self.block_size]
        y = mask[offset+1:offset+self.block_size+1]
        
        return x.long(), y.long()

# In[10]:


class MaskPredFormer(pl.LightningModule):
    def __init__(self, predictor, batch_size, lr, max_epochs):
        super().__init__()
        self.predictor = predictor
        self.batch_size = batch_size
        self.lr = lr
        self.max_epochs=max_epochs
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, 
            num_workers=4, shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            val_set, batch_size=1, 
            num_workers=4, shuffle=False, pin_memory=True
        )

    def common_step(self, batch):
        x, y = batch
        y_hat = self.predictor(x) # (b, t, 49, h, w)

        # compute loss
        b, t, *_ = y.shape
        assert b == y_hat.shape[0] and t == y_hat.shape[1], "Batch size or sequence length mismatch"

        y = y.view(b*t, *y.shape[2:]) # (b*t, h, w)
        y_hat = y_hat.view(b*t, *y_hat.shape[2:]) # (b*t, 49, h, w)

        loss = torch.nn.functional.cross_entropy(y_hat, y)

        # return pred
        y_hat_pred = torch.argmax(y_hat, dim=1).view(b, t, *y_hat.shape[2:])        
        return loss, y_hat_pred
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _ = self.common_step(batch)
        self.log('val_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=(len(train_set)//self.batch_size)*self.max_epochs, eta_min=1e-5)
        return [optim], [sch]

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
        
        x, y = self.val_dataset[random.randint(0, len(self.val_dataset))]
        x = x.unsqueeze(0).to(pl_module.device)

        # forward pass
        _, pred = pl_module.common_step(x)
        x = x.detach().cpu().squeeze(0)

        # TODO: fix video generation here, put only autoregressive generation

        # # autoregressive generation
        # pred_ar = sample[:, :11]
        # for t in range(11):
        #     _, pred_t, _ = pl_module.common_step(pred_ar)
        #     pred_ar = torch.cat([pred_ar, pred_t[:, -1:]], dim=1)
        # pred_ar = pred_ar.detach().cpu().squeeze(0)
        
        # # sample and log videos
        # pred_imgs = []
        # pred_ar_imgs = []
        # gt_imgs = []
        # for t in range(pred.shape[0]):
        #     pred_img = self.apply_cm(torch.argmax(pred[t], dim=0))
        #     pred_ar_img = self.apply_cm(torch.argmax(pred_ar[t], dim=0))
        #     gt_img = self.apply_cm(gt[t])
        #     pred_imgs.append(pred_img)
        #     pred_ar_imgs.append(pred_ar_img)
        #     gt_imgs.append(gt_img)

        # pred_imgs = np.stack(pred_imgs, axis=0)
        # pred_ar_imgs = np.stack(pred_ar_imgs, axis=0)
        # gt_imgs = np.stack(gt_imgs, axis=0)
        # video = (np.concatenate([gt_imgs, pred_imgs, pred_ar_imgs], axis=3) * 255).astype(np.uint8)
        # trainer.logger.experiment.log({
        #     "val_video": wandb.Video(video, fps=4, format="gif")
        # })
        
    


# In[11]:
if __name__ == "__main__":
    pl.seed_everything(42)
    predictor = Predictor()
    predictor.cuda();


    print("Loading datasets..")
    train_set = MaskPredDataset("data_prepared", "train") # MemMapDataset("data_prepared", "train", length=1000)
    print("INFO: Train set has", len(train_set))
    val_set = MaskPredDataset("data_prepared", "val")
    print("INFO: Val set has", len(val_set))

    model = MaskPredFormer(predictor, batch_size=16, lr=0.01737, max_epochs=20)

    logger = WandbLogger(project="mask-predformer")
    sample_video_cb = SampleVideoCallback(val_set)
    checkpoint_callback = ModelCheckpoint(dirpath="models/", 
                                        filename='{epoch}-{val_loss:.3f}',
                                        monitor='val_loss', save_top_k=5, mode='min', save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(max_epochs=model.max_epochs, accelerator="gpu", devices=1, logger=logger, 
                        callbacks=[sample_video_cb, checkpoint_callback, lr_monitor])
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(MaskPredFormer(predictor, batch_size=1), mode="binsearch")
    # lr_finder = tuner.lr_find(model)
    # # Results can be found in
    # print(lr_finder.results)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    # plt.savefig("lr_finder.png")
    # In[12]:

    trainer.fit(
        model,
        ckpt="models/last.ckpt"
    )


