import os

import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.tuner import Tuner

from maskpredformer.autoregressive_trainer import MaskSimVPAutoRegressiveModule, SampleAutoRegressiveVideoCallback

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
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # Trainer arguments
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Hyperparameters for the model
    parser.add_argument("--simvp_path", type=str, required=True, help="checkpoint path to pretrained simvp prior")
    parser.add_argument("--backprop_indices", type=int, default=[10], nargs="+", help="steps to backprop from in autoregressive training")
    parser.add_argument("--unlabeled", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=10)

    # MultiGPU
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default='auto')

    args = parser.parse_args()

    pl.seed_everything(42)
    
    mask_sim_vp_ckpt = torch.load(args.simvp_path)

    autoregressive_params = mask_sim_vp_ckpt['hyper_parameters']
    autoregressive_params['unlabeled'] = args.unlabeled
    autoregressive_params['backprop_indices'] = args.backprop_indices
    autoregressive_params['batch_size'] = args.batch_size
    autoregressive_params['lr'] = args.lr
    autoregressive_params['max_epochs'] = args.max_epochs
    
    module = MaskSimVPAutoRegressiveModule(**autoregressive_params)
    module.load_state_dict(mask_sim_vp_ckpt["state_dict"])
    print("INFO: loaded model checkpoint from MaskSimVP")

    run_name = dict_to_folder_name({
        "prefix": "AR",
        "simvp": os.path.basename(args.simvp_path),
        "backprop_indices": autoregressive_params['backprop_indices'],
        "unlabeled": autoregressive_params['unlabeled']
    })
    dirpath = os.path.join("checkpoints/", run_name)

    sample_video_cb = SampleAutoRegressiveVideoCallback(
        module.val_set, video_path=os.path.join(dirpath, "val_videos")
    )
    logger = WandbLogger(project="mask-predformer", name="gSTA_AR")
    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="simvp_ar_{epoch}-{val_loss:.3f}",
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    trainer = pl.Trainer(
        max_epochs=module.hparams.max_epochs,
        accelerator="gpu",
        devices=args.devices,
        strategy=args.strategy,
        logger=logger,
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=0.25,
        callbacks=[sample_video_cb, checkpoint_callback, lr_monitor],
    )
    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(module, mode="power")

    ckpt_path = os.path.join(dirpath, "last.ckpt")
    trainer.fit(module, ckpt_path=ckpt_path if os.path.exists(ckpt_path) else None)
