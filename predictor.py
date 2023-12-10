import torch
import torch.nn as nn
import torch.nn.functional as F

#@title Conv Block
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
    
#@title Encoder Block
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)  
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
#@title Decoder Block
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x
    
class Predictor(nn.Module):
    def __init__(self, 
                embed_dim=32, 
                channels=[32, 64, 128], 
                bottleneck_dim=128,
                transformer_dim=256,
                nhead=8,
                transformer_num_layers=6,
                block_size=11,
                out_channels=49
        ):
        super(Predictor, self).__init__()
        self.embed_dim = embed_dim
        self.bottleneck_dim = bottleneck_dim
        self.channels = channels
        self.block_size = block_size
        
        
        """ Embeddings """
        self.token_embeddings = nn.Embedding(49, embed_dim)
        self.position_embeddings = nn.Embedding(block_size, transformer_dim)

        """ Transformer Encoder """
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim, 
                                       nhead=nhead, batch_first=True), 
            num_layers=transformer_num_layers
        )
        
        """ Downscale Conv """
        self.init_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, stride=2)

        self.proj = nn.Sequential(
            nn.Linear(bottleneck_dim*10*15, bottleneck_dim*10), 
            nn.LeakyReLU(), 
            nn.Linear(bottleneck_dim*10, transformer_dim))
        self.up_proj = nn.Sequential(
            nn.Linear(transformer_dim, bottleneck_dim*10),
            nn.LeakyReLU(),
            nn.Linear(bottleneck_dim*10, bottleneck_dim*10*15))
        
        
        """ Encoder """
        self.encoder_blocks = nn.ModuleList(
            [encoder_block(embed_dim, channels[0])] + [
                encoder_block(channels[i], channels[i+1]) for i in range(len(channels)-1)
            ]
        )

        """ Bottleneck """
        self.b = conv_block(channels[-1], bottleneck_dim)

        """ Decoder """
        self.decoder_blocks = nn.ModuleList([
            decoder_block(bottleneck_dim, channels[-1])] + [
            decoder_block(channels[i], channels[i-1]) for i in range(len(channels)-1, 0, -1)
        ])
        
        """ UpConv """
        self.up_conv = nn.ConvTranspose2d(channels[0], channels[0], kernel_size=2, stride=2, padding=0)

        """ Output """
        self.out = nn.Conv2d(channels[0], out_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Embed input labels """
#         import pdb; pdb.set_trace()
        embeds = self.token_embeddings(inputs) # (b, t, h, w) -> (b, t, h, w, embed_dim)
        embeds = embeds.permute(0, 1, 4, 2, 3) # (b, t, h, w, embed_dim) -> (b, embed_dim, t, h, w)
        b, t, c, h, w = embeds.shape
        x = embeds.contiguous().view(b*t, c, h, w) # (b, t, c, h, w) -> (b*t, c, h, w)
        
        """ Downscale x2 """
        x = self.init_conv(x)
        
        """ Encoder """
        encoder_features = []        

        for encoder_block in self.encoder_blocks:
            skip, x = encoder_block(x)
            encoder_features.append(skip)
    
        """ Bottleneck """
        x = self.b(x)

        """ Proj to Transformer """
        _, _, h_b, w_b = x.shape
        x = x.view(b*t, -1) # (b*t, bottleneck_dim, h_b, w_b) -> (b*t, bottleneck_dim*h_b*w_b)
        x = self.proj(x) # (b*t, bottleneck_dim*h_b*w_b) -> (b*t, transformer_dim)

        """ Transformer encoder """

        # add positional embeddings
        x = x.view(b, t, -1) # (b*t, transformer_dim) -> (b, t, transformer_dim)
        pos_emb = self.position_embeddings(torch.arange(t).to(x.device)).unsqueeze(0) # (1, t, transformer_dim)
        x = x + pos_emb

        # causal attention processing
        x = self.transformer_encoder(x, 
                                     is_causal=True, 
                                     mask=nn.Transformer.generate_square_subsequent_mask(t).to(x.device)
                                   ) # (b, t, transformer_dim) -> (b, t, transformer_dim)

        """ Proj to Decoder """
        x = x.view(b*t, -1)
        x = self.up_proj(x) # (b*t, transformer_dim) -> (b*t, bottleneck_dim*h_b*w_b)

        """ Decoder """
        x = x.view(b*t, self.bottleneck_dim, h_b, w_b) # (b*t, bottleneck_dim*h_b*w_b) -> (b*t, bottleneck_dim, h_b, w_b)
        for decoder_block, skip in zip(self.decoder_blocks, encoder_features[::-1]):
            x = decoder_block(x, skip)
        
        """ UpConv """
        x = self.up_conv(x)

        """ Output """
        x = self.out(x)
        x = x.view(b, t, *x.shape[1:])

        return x


if __name__ == "__main__":
    b = 1
    t = 22
    h = 160
    w = 240
    predictor = Predictor()
    mask_predictions = torch.randn(b, t, 49, h, w)
    print(predictor(mask_predictions).shape)
