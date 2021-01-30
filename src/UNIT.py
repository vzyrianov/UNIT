import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
 
            nn.Linear(hidden_dim, hidden_dim), #ADDED THIS
            nn.GELU(),
            nn.Dropout(dropout),


            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(
                    dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

#'''
class SeTr(nn.Module):
    #dim -> Embedding Dimension
    #depth -> Number of Transformer Layers
    #heads -> Heads
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dropout=0., emb_dropout=0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.channels = channels

        #assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        self.image_size = image_size
        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()
        

        mlp_input_dimension = dim * (num_patches) + (image_size * image_size * channels)
        mlp_inner_dimension = dim * (num_patches) + (image_size * image_size * channels)
        mlp_output_dimension = (image_size * image_size * channels)
        self.mlp_head = nn.Sequential(
            #nn.Linear(mlp_input_dimension, mlp_inner_dimension),
            #nn.GELU(),

            #nn.LayerNorm(mlp_inner_dimension),
            #nn.Linear(mlp_inner_dimension, mlp_output_dimension)

            nn.Linear(mlp_input_dimension, mlp_output_dimension)
        )

    def forward(self, img, mask=None):
        p = self.patch_size
        b = img.shape[0]
        x = rearrange(
            img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        #cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        #x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n)]# + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = rearrange(x[:, :], 'b p d -> b (p d)')
        img = rearrange(img, 'b c w h -> b (c w h)')
        combined = torch.cat((x, img), 1)
        x = torch.reshape(self.mlp_head(combined), (b, self.channels, self.image_size, self.image_size))
        return x

def double_conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out_channels)
    )

class UNIT(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.setr = SeTr(image_size=10, patch_size=2, dim=8, depth=2, heads=8, mlp_dim=10, channels=64)

        self.dconv_down1 = double_conv(4, 32, 2)
        self.dconv_down2 = double_conv(32, 64)
        #self.dconv_down3 = double_conv(64, 128)
        #self.dconv_down4 = double_conv(128, 256)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        #self.dconv_up3 = double_conv(256, 128)
        #self.dconv_up2 = double_conv(128 + 64, 64)
        self.dconv_up2 = double_conv(128, 64)
        self.dconv_up1 = double_conv(64 + 32, 32) 
        
        self.conv_last = nn.Conv2d(32, 4, 1)
        
        
    def forward(self, x):
        
        # Down
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
#        conv3 = self.dconv_down3(x)
#        x = self.maxpool(conv3)   
        
        #conv4 = self.dconv_down4(x)
        #x = self.maxpool(conv4)


        # Middle    
        x = self.setr(x)

        #Up
#        x = self.upsample(x)
#        x = torch.cat([x, conv3], dim=1)

#        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        #x = self.upsample(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.upsample(x)
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out