# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_mask_simvp.ipynb.

# %% auto 0
__all__ = ['DEFAULT_MODEL_CONFIG', 'MaskSimVP']

# %% ../nbs/00_mask_simvp.ipynb 2
from openstl.models.simvp_model import SimVP_Model
import torch.nn as nn

# %% ../nbs/00_mask_simvp.ipynb 3
DEFAULT_MODEL_CONFIG = {
    # For MetaVP models, the most important hyperparameters are: 
    # N_S, N_T, hid_S, hid_T, model_type
    'in_shape': [11, 32, 160, 240],
    'hid_S': 64,
    'hid_T': 256,
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

    def forward(self, tokens):
        x = self.token_embeddings(tokens)
        x = x.permute(0, 1, 4, 2, 3)
        x = self.simvp(x)

        b, t, c, h, w = x.shape
        x = x.view(b*t, c, h, w)

        x = self.out_conv(x)
        x = x.view(b, t, 49, h, w)
        return x


