import torch
import os
import tqdm

data_path="data_prepared/val"
all_videos = [os.path.join(data_path, name) for name in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, name))]
all_videos.sort(key= lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))

all_masks = []
for video in tqdm.tqdm(all_videos):
    all_masks.append(torch.load(os.path.join(video, "mask.pt")))
all_masks = torch.stack(all_masks, dim=0)

torch.save(all_masks, os.path.join(data_path, "val_masks.pt"))
