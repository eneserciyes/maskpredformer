# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_mask_simvp.ipynb.

# %% auto 0
__all__ = ['DEFAULT_MODEL_CONFIG', 'MaskSimVP']

# %% ../nbs/00_mask_simvp.ipynb 2
from openstl.models.simvp_model import SimVP_Model
import torch.nn as nn
import torch

# %% ../nbs/00_mask_simvp.ipynb 3
DEFAULT_MODEL_CONFIG = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'in_shape': [11, 3, 160, 240],
    'hid_S': 64,
    'hid_T': 512,
    'N_S': 4,
    'N_T': 8,
    'model_type': 'gSTA',
}

# %% ../nbs/00_mask_simvp.ipynb 4
class MaskSimVP(nn.Module):
    def __init__(self, in_shape, hid_S, hid_T, N_S, N_T, model_type):
        super().__init__()
        c = in_shape[1]
        self.simvp = SimVP_Model(
            in_shape=in_shape, hid_S=hid_S, 
            hid_T=hid_T, N_S=N_S, N_T=N_T, 
            model_type=model_type)
        self.token_embeddings = nn.Embedding(49, c)
        self.out_conv = nn.Conv2d(c, 49, 1, 1)
        self.pre_seq_len = 11
        self.after_seq_len = 11

    def forward(self, tokens):
        x = self.token_embeddings(tokens)
        x = x.permute(0, 1, 4, 2, 3)

        d = self.after_seq_len // self.pre_seq_len
        m = self.after_seq_len % self.pre_seq_len

        y_hat = []
        cur_seq = x.clone()
        for _ in range(d):
            cur_seq = self.simvp(cur_seq)
            y_hat.append(cur_seq)
        
        if m != 0:
            cur_seq = self.simvp(cur_seq)
            y_hat.append(cur_seq[:, :m])
        
        y_hat = torch.cat(y_hat, dim=1)

        b, t, c, h, w = y_hat.shape
        y_hat = y_hat.view(b*t, c, h, w)

        y_hat_logits = self.out_conv(y_hat)
        y_hat_logits = y_hat_logits.view(b, t, 49, h, w)
        return y_hat_logits
