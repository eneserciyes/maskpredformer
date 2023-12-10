import os

import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from maskpredformer.trainer import MaskSimVPModule, SampleVideoCallback
from maskpredformer.mask_simvp import DEFAULT_MODEL_CONFIG


def list_to_folder_name(l):
    return "-".join([str(x) for x in l])


def dict_to_folder_name(d):
    return "_".join(
        [
            f"{k}={list_to_folder_name(v) if isinstance(v, list) else v}"
            for k, v in d.items()
        ]
    )


if __name__ == "__main__":
    pl.seed_everything(42)
    module = MaskSimVPModule(
        **DEFAULT_MODEL_CONFIG,
        batch_size=1,
        lr=1e-3,
        weight_decay=0.0,
        max_epochs=100,
        data_root="data/DL/",
    )
    run_name = dict_to_folder_name(module.hparams)
    dirpath = os.path.join("checkpoints/", run_name)

    sample_video_cb = SampleVideoCallback(module.val_set)
    logger = WandbLogger(project="mask-predformer", name="train_simvp")
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="simvp_{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=5,
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=module.hparams.max_epochs,
        accelerator="gpu",
        logger=logger,
        log_every_n_steps=100,
        val_check_interval=0.5,
        gradient_clip_val=0.5,
        callbacks=[sample_video_cb, checkpoint_callback, lr_monitor],
    )

    ckpt_path = os.path.join(dirpath, "last.ckpt")
    trainer.fit(module, ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None)
