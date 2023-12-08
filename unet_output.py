import torch 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os 
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt
import torch
from torchmetrics import JaccardIndex

model = torch.jit.load('mask_unet/unet9.pt')

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)


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

            if self.train_or_val != "hidden":
                mask = np.load(mask_path)[frame] 
                masks.append(torch.from_numpy(mask).long())
        
        if self.train_or_val != "hidden":
            return torch.stack(images), torch.stack(masks)
        else:
            return torch.stack(images), torch.zeros((1, 160, 240)).long()

    def __len__(self):
        return len(os.listdir(self.data_path))

transform = transforms.Compose([
    transforms.ToTensor(),     
    transforms.Resize((160, 240), antialias=True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

data_path = 'Dataset_Student/'

dataset = WenmaSet(data_path = data_path + 'val', 
                      train_or_val = 'val', 
                      transform = transform)

dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=4)

import pdb
pdb.set_trace()
all_preds = []
all_masks = []
for i in tqdm.tqdm(range(len(dataset))):
    images, masks = dataset[i]
    images = images.to(device)
    masks = masks.to(device)
    with torch.cuda.amp.autocast():
        mask_prediction = model(images)
        mask_prediction = torch.argmax(mask_prediction, dim=1)
        all_preds.append(mask_prediction.detach().cpu().numpy())
        all_masks.append(masks.detach().cpu().numpy())

all_masks_final = np.stack([mask[-1] for mask in all_masks])
all_preds_final = np.stack([pred[-1] for pred in all_preds])

print(np.unique(all_masks_final))
# np.save('/vast/me2646/all_masks_final_val_unet9.npy', all_masks_final)
# np.save('/vast/me2646/all_preds_final_val_unet9.npy', all_preds_final)
# np.save('/vast/me2646/all_images_final_val_unet9.npy', all_images_final)

jaccard = JaccardIndex(task='multiclass', num_classes=49)
print("IOU score:", jaccard(torch.from_numpy(all_preds_final), torch.from_numpy(all_masks_final)))
