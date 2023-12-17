# Team 19 - We Need More Than Attention (WENMA)

Placed 2nd out of 38 teams in NYU Deep Learning final project in Fall 2023.

## Instructions

0. Place the data (or symlink) it to `data/Dataset_Student`. This directory should contain `train`, `val` and `unlabeled` folders.

1. The first step is to train a UNet with these data.
#TODO: talk about UNet training

2. Now, we can generate masks from the data.

- To generate masks for any split, run `python generate_masks.py --model_checkpoint <path_to_checkpoint> --data_root <data_root> --split <split_name> --output_file <output_file>`
- For training on the labeled data only, you can run for `train` and `val` splits.

3. Now, we can train our prediction model on the generated masks.

- For training only on labeled set:
`python3 train_simvp.py --downsample --in_shape 11 49 160 240 --lr 1e-3 --pre_seq_len=11 --aft_seq_len=1 --max_epochs 20 --batch_size 4 --check_val_every_n_epoch=1`

- To train on labeled and unlabeled set, generate masks for unlabeled and add `--unlabeled` flag.

4. Now, we can finetune with scheduled sampling.

`python3 train_simvp_ss.py --simvp_path checkpoints/simvp_epoch=16-val_loss=0.014.ckpt --sample_step_inc_every_n_epoch 20 --max_epochs 100 --batch_size 4 --check_val_every_n_epoch 2`

We used the checkpoint after second epoch of scheduled sampling for our final submission.
