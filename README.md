# Team 19 - We Need More Than Attention (WENMA)

Placed 2nd out of 38 teams in NYU Deep Learning final project in Fall 2023.

## Instructions

0. Place the data (or symlink) it to `data/Dataset_Student`. This directory should contain `train`, `val` and `unlabeled` folders.

1. The first step is to train a UNet with these data: `python3 train_unet.py`. This will save the model in `checkpoints/unet9.pt`.

2. Now, we can generate masks from the data.

- To generate masks for train and val splits, run 
`python generate_masks.py --model_checkpoint checkpoints/unet9.pt --data_root data/Dataset_Student --split <train, val> --output_file <data/DL/train_masks.pt, data/DL/val_masks.pt>`
- For training the world model on the labeled data only, you can run for `train` and `val` splits.

- For this step, we also need to merge all ground truth masks into one file. To do this, run 
`python merge_masks.py --data_root data/Dataset_Student --split <train, val> --output_file <data/DL/train_gt_masks.pt, data/DL/val_gt_masks.pt>` for `train` and `val` splits.

Or, you can get the pre-generated masks from here (this link requires an NYU account):
| Split | Link |
| ------------- | ------------- |
| Train  | [Link](https://drive.google.com/file/d/1T3tFfziIjQhSiwSEaJJQSx11x6MOJmla/view?usp=sharing)  |
| Validation  | [Link](https://drive.google.com/file/d/1FGxuEG-IZdVe3dDPE1AKj0BYn4ys3g_t/view?usp=sharing) |

3. Now, we can train our prediction model on the generated masks.

- For training only on labeled set:
`python3 train_simvp.py --downsample --in_shape 11 49 160 240 --lr 1e-3 --pre_seq_len=11 --aft_seq_len=1 --max_epochs 20 --batch_size 4 --check_val_every_n_epoch=1`

- To train on labeled and unlabeled set, generate masks for unlabeled and add `--unlabeled` flag.

4. Now, we can finetune with scheduled sampling.

`python3 train_simvp_ss.py --simvp_path checkpoints/simvp_epoch=16-val_loss=0.014.ckpt --sample_step_inc_every_n_epoch 20 --max_epochs 100 --batch_size 4 --check_val_every_n_epoch 2`

We used the checkpoint after second epoch of scheduled sampling for our final submission. The checkpoint is [here]().

**Checkpoints**
| Name | Link |
| ------------- | ------------- |
| Best w/o scheduled sampling  | Content Cell  |
| Best after  | Content Cell  |


5. To generate predictions on the hidden set, run this notebook: `nbs/96_get_results_combined_with_unet.ipynb`

The final predictions are [here]().
