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
                in_channels=49, 
                channels=[64, 128, 256, 512], 
                bottleneck_dim=512,
                nhead=8,
                transformer_num_layers=6,
        ):
        super(Predictor, self).__init__()
        self.in_channels = in_channels
        self.bottleneck_dim = bottleneck_dim
        self.channels = channels

        """ Transformer Encoder """
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=bottleneck_dim, nhead=nhead, batch_first=True), 
            num_layers=transformer_num_layers
        )

        self.proj = nn.Linear(bottleneck_dim*10*15, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, bottleneck_dim*10*15)
        
        """ Encoder """
        self.encoder_blocks = nn.ModuleList([encoder_block(in_channels, channels[0])] + [
            encoder_block(channels[i], channels[i+1]) for i in range(len(channels)-1)
        ])

        """ Bottleneck """
        self.b = conv_block(channels[-1], bottleneck_dim)

        """ Decoder """
        self.decoder_blocks = nn.ModuleList([
            decoder_block(bottleneck_dim, channels[-1])] + [
            decoder_block(channels[i], channels[i-1]) for i in range(len(channels)-1, 0, -1)
        ])

        """ Output """
        self.out = nn.Conv2d(channels[0], in_channels, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        encoder_features = []
        b, t, c, h, w = inputs.shape
        x = inputs.contiguous().view(b*t, c, h, w) # (b, t, c, h, w) -> (b*t, c, h, w)
        for encoder_block in self.encoder_blocks:
            skip, x = encoder_block(x)
            encoder_features.append(skip)
    
        """ Bottleneck """
        x = self.b(x)

        """ Proj to Transformer """
        _, _, h_b, w_b = x.shape
        x = x.view(b*t, -1) # (b*t, bottleneck_dim, h_b, w_b) -> (b*t, bottleneck_dim*h_b*w_b)
        x = self.proj(x) # (b*t, bottleneck_dim*h_b*w_b) -> (b*t, bottleneck_dim)

        """ Transformer encoder """

        x = x.view(b, t, -1) # (b*t, bottleneck_dim) -> (b, t, bottleneck_dim)
        x = self.transformer_encoder(x, 
                                     is_causal=True, 
                                     mask=nn.Transformer.generate_square_subsequent_mask(t).to(x.device)) # (b, t, bottleneck_dim) -> (b, t, bottleneck_dim

        """ Proj to Decoder """
        x = x.view(b*t, -1)
        x = self.up_proj(x) # (b*t, bottleneck_dim) -> (b*t, bottleneck_dim*h_b*w_b)

        """ Decoder """
        x = x.view(b*t, self.bottleneck_dim, h_b, w_b) # (b*t, bottleneck_dim*h_b*w_b) -> (b*t, bottleneck_dim, h_b, w_b)
        for decoder_block, skip in zip(self.decoder_blocks, encoder_features[::-1]):
            # add the connections from previous frame to current frame
            # skip = skip.view(b, t, *skip.shape[1:])
            # skip = torch.cat([skip[:, 0:1], skip[:, :-1]], dim=1).view(b*t, *skip.shape[2:])
            x = decoder_block(x, skip)

        """ Output """
        x = self.out(x)
        x = x.view(b, t, *x.shape[1:])

        return x


if __name__ == "__main__":
    b = 2
    t = 22
    h = 160
    w = 240
    predictor = Predictor()
    mask_predictions = torch.randn(b, t, 49, h, w)
    print(predictor(mask_predictions).shape)
