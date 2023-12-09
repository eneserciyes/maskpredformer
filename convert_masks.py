#!/usr/bin/env python
# coding: utf-8

# In[22]:


import os
from pathlib import Path
import torch
import tqdm


# In[26]:


mode = "unlabeled"
data_root = Path("data_prepared") / mode


# In[27]:


all_videos = [os.path.join(data_root, name) for name in 
 os.listdir(data_root) if os.path.isdir(os.path.join(data_root, name))]


# In[28]:
print(all_videos[:10])

# all_videos[:10]


# In[29]:


for video in tqdm.tqdm(all_videos):
    masks_old = torch.load(os.path.join(video, "mask.pt"))
    if masks_old.shape[1] != 49: continue 
    masks = torch.argmax(masks_old, dim=1).to(torch.uint8)

    assert torch.allclose(torch.argmax(masks_old, dim=1), masks.long())
    # masks_gt = torch.load(os.path.join(video, "mask_gt.pt"))
    # break
    torch.save(masks, os.path.join(video, "mask.pt"))

