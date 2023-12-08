import torch 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os 
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt
import torch
import os
import time
import re
import shutil


device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.jit.load('models/unet9.pt')
model.to(device);

class WenmaSet(Dataset):
    def __init__(self, data_path, train_or_val, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.train_or_val = train_or_val

    def __getitem__(self, ind):
        if self.train_or_val == "train":
            pass
        elif self.train_or_val == "val": 
            ind = ind + 1000
        elif self.train_or_val == "unlabeled":
            ind = ind + 2000
        elif self.train_or_val == "hidden":
            ind = ind + 15000

        video_path = os.path.join(self.data_path, 'video_{}'.format(ind))
        mask_path = os.path.join(video_path, 'mask.npy')
        
        images = []
        masks = []

        for frame in range(0, 22):
            image_path = os.path.join(video_path, 'image_{}.png'.format(frame))
            image = np.array(Image.open(image_path))

            if self.transform:
                image = self.transform(image)
            images.append(image)

            if self.train_or_val in ["train", "val"]:
                mask = np.load(mask_path)[frame] 
                masks.append(torch.from_numpy(mask).long())
        
        if self.train_or_val in ["train", "val"]:
            return torch.stack(images), torch.stack(masks)
        else:
            return torch.stack(images), None

    def __len__(self):
        return len(os.listdir(self.data_path))
    
transform = transforms.Compose([
    transforms.ToTensor(),     
    transforms.Resize((160, 240), antialias=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

data_path = '/dataset/'
mode = 'unlabeled'

dataset = WenmaSet(data_path = data_path + mode, 
                      train_or_val = mode, 
                      transform = transform)

save_root = os.path.join('/vast/me2646/data_prepared/', mode)
os.makedirs(save_root, exist_ok=True)
for i in tqdm.tqdm(range(len(dataset))):
    if i < 1778:
        continue
#     import pdb; pdb.set_trace()
    offset = 0 if mode=="train" else 1000 if mode=="val" else 2000 if mode == "unlabeled" else 15000
    save_path = os.path.join(save_root, f"video_{i+offset}")
    os.makedirs(save_path, exist_ok=True)
    images, masks = dataset[i]
    images = images.to(device)
    with torch.cuda.amp.autocast():
        mask_prediction = model(images)
        mask_prediction = mask_prediction.detach().cpu()
    
    torch.save(mask_prediction, os.path.join(save_path, "mask.pt"))
    if mode in ["train", "val"]:
        torch.save(masks, os.path.join(save_path, "mask_gt.pt"))