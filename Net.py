import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def pixel_upsample(x, H, W):
    B, N, C = x.shape
    assert N == H*W
    x = x.reshape(B, H, W, C)
    x = F.pixel_shuffle(x, 2)
    B, H, W, C = x.shape
    return x, H, W, C

class SMLayerNormalization(nn.Module):
    def __init__(self, eps=1e-6, hidden_size=None):
        super().__init__()
        self.eps = eps
        if hidden_size is not None:
            self.hidden_size = hidden_size
        self.h = nn.Linear(self.hidden_size, self.hidden_size)
        self.gamma = nn.Linear(self.hidden_size, self.hidden_size) 
        self.beta = nn.Linear(self.hidden_size, self.hidden_size)
        self.ln = nn.LayerNorm(self.hidden_size, eps=self.eps)

    def forward(self, x, z):
        x = self.ln(x)
        h = F.relu(self.h(z))
        scale = self.gamma(h)
        shift = self.beta(h)
        x = x * scale[:, None, :] + shift[:, None, :]
        return x

class AdditiveAttention(nn.Module):
    def __init__(self, model_dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.model_dim = model_dim
        assert model_dim % n_heads == 0
        self.depth = model_dim // n_heads

        self.wq = nn.Linear(model_dim, model_dim)
        self.wk = nn.Linear(model_dim, model_dim)
        self.wv = nn.Linear(model_dim, model_dim)
        
        self.q_attn = nn.Linear(model_dim, n_heads)
        self.out = nn.Linear(model_dim, model_dim)
    
    def split_heads(self, x):
        B, N, D = x.shape
        x = x.reshape(B, N, self.n_heads, -1).transpose(1, 2)
        return x
    
    def forward(self, q, k, v):
        B, N, D = q.shape
        q = self.wq(q)
        k = self.wk(k)  
        v = self.wv(v)

        attn = self.q_attn(q).transpose(-1, -2) / math.sqrt(self.depth)
        attn = F.softmax(attn, dim=-1)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        global_q = torch.einsum('b h n, b h n d -> b h d', attn, q)
        global_q = global_q.unsqueeze(2)

        p = global_q * k
        r = p * v
        
        r = r.transpose(1, 2).reshape(B, N, D)
        output = self.out(r)
        return output, attn

class SMLadaformer(nn.Module):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, dropout=0.0, eps=1e-6):
        super().__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim)
        )
        self.norm1 = SMLayerNormalization(eps, model_dim)
        self.norm2 = SMLayerNormalization(eps, model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, z):
        x_norm1 = self.norm1(x, z)
        attn_out, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_out = x + self.dropout1(attn_out)
        
        x_norm2 = self.norm2(attn_out, z)
        mlp_out = self.mlp(x_norm2)
        return self.dropout2(mlp_out), attn_maps

class PositionalEmbedding(nn.Module):
    def __init__(self, n_patches, model_dim):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, n_patches, model_dim))

    def forward(self, x):
        return x + self.pos_emb

class Generator(nn.Module):
    def __init__(self, img_size=32, model_dim=[1024, 256, 64], heads=[2, 2, 2], 
                 mlp_dim=[2048, 1024, 512], dec_dim=[]):
        super().__init__()

        self.init = nn.Linear(128, 8*8*model_dim[0])
        self.pos_emb_8 = PositionalEmbedding(64, model_dim[0])
        self.block8 = SMLadaformer(model_dim[0], heads[0], mlp_dim[0])
        self.conv8 = nn.Conv2d(model_dim[0], model_dim[1], 3, padding=1)

        self.pos_emb_16 = PositionalEmbedding(256, model_dim[1])
        self.block16 = SMLadaformer(model_dim[1], heads[1], mlp_dim[1])
        self.conv16 = nn.Conv2d(model_dim[1], model_dim[2], 3, padding=1)

        self.pos_emb_32 = PositionalEmbedding(1024, model_dim[2])  
        self.block32 = SMLadaformer(model_dim[2], heads[2], mlp_dim[2])

        self.dec_dim = dec_dim
        if dec_dim:
            self.dec = nn.Sequential(
                *[nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                nn.Conv2d(dim, dim, 3, padding=1),
                                nn.BatchNorm2d(dim),
                                nn.LeakyReLU(0.2)) 
                  for dim in dec_dim])
        else:
            self.patch_size = img_size // 32
        
        self.out = nn.Conv2d(model_dim[2], 3, 3, padding=1)

    def forward(self, z):
        B = z.shape[0]
        x = self.init(z).reshape(B, 64, -1)
        x = self.pos_emb_8(x)
        x, attn8 = self.block8(x, z)

        x, H, W, C = pixel_upsample(x, 8, 8)
        x = self.conv8(x)
        x = x.reshape(B, H*W, C)

        x = self.pos_emb_16(x)
        x, attn16 = self.block16(x, z)

        x, H, W, C = pixel_upsample(x, H, W)
        x = self.conv16(x)
        x = x.reshape(B, H*W, C)

        x = self.pos_emb_32(x)  
        x, attn32 = self.block32(x, z)

        x = x.reshape(B, 32, 32, -1)

        if self.dec_dim:
            x = self.dec(x)
        elif self.patch_size != 1:
            x = F.pixel_shuffle(x, self.patch_size)
            
        return self.out(x), [attn8, attn16, attn32]

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        self.direct = nn.Sequential(
           nn.AvgPool2d(kernel_size=stride, stride=stride),
           nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return (self.main(x) + self.direct(x)) / 2

class Ladaformer(nn.Module):
    def __init__(self, model_dim, n_heads=2, mlp_dim=512, dropout=0.0, eps=1e-6):
        super().__init__()
        self.attn = AdditiveAttention(model_dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim, eps=eps)
        self.norm2 = nn.LayerNorm(model_dim, eps=eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x_norm1 = self.norm1(x)

        attn_out, attn_maps = self.attn(x_norm1, x_norm1, x_norm1)
        attn_out = x + self.dropout1(attn_out)

        x_norm2 = self.norm2(attn_out)
        mlp_out = self.mlp(x_norm2)
        return self.dropout2(mlp_out) + attn_out, attn_maps
    
class Discriminator(nn.Module):
    def __init__(self, img_size=32, enc_dim=[64, 128, 256], out_dim=[512, 1024], 
                 mlp_dim=512, heads=2):
        super().__init__()
        assert len(enc_dim) == 2 or 3 or 4 or 5
        self.enc_dim = enc_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, enc_dim[0], 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2)
        )
        self.encoder = nn.ModuleList([
            DownBlock(enc_dim[i], enc_dim[i+1]) for i in range(len(enc_dim)-1)
        ])

        self.pos_emb_8 = PositionalEmbedding(256, enc_dim[-1])
        self.block8 = Ladaformer(enc_dim[-1], heads, mlp_dim)

        self.conv4 = nn.Conv2d(enc_dim[-1], out_dim[0], 3, 1, 1)
        self.down4 = nn.Sequential(
            nn.Conv2d(out_dim[0], out_dim[1], 1, 1, 0, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_dim[1], 1, 4, 1, 0)  
        )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Identity()
        )

    def forward(self, x):
        x = self.conv1(x)

        for down in self.encoder:
            x = down(x)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H*W).transpose(-1, -2)
        
        x = self.pos_emb_8(x)
        x, maps16 = self.block8(x)

        x = x.transpose(-1, -2).reshape(B, C, H, W)
        x = F.pixel_unshuffle(x, 2)
        x = self.conv4(x)
        x = self.down4(x)

        return self.out(x)