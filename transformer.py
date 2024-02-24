import torch
import numpy as np
import pandas as pd
import scanpy as sc
import torch.nn.functional as F
from easydl import *
from anndata import AnnData
from torch import nn, einsum
from scipy.stats import pearsonr
from torch.autograd import Function
from torch.autograd.variable import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
    # @get_local('attn')
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads  # 前面加一个星号（*）表示对元组、列表或其他可迭代对象进行解包，将其拆分成单独的元素
        # print("attention:", x.shape)  [1, 295, 1024]
        qkv = self.to_qkv(x).chunk(3, dim = -1) # 对张量进行切分操作, return list, 每个和原来尺寸相同
        # print(qkv[0].shape)   [1, 295, 1024]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        # print("QKV:",q.shape, k.shape, v.shape)  # [1, 16, 295, 64])
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # 计算多线性表达式的方法（即乘积和） q x k
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v) # asttention 分配
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class attn_block(nn.Module):
    #  dim, 16, 64, 1024, dropout
    def __init__(self, dim=1024, heads=8, dim_head=64, mlp_dim=1024, dropout = 0.):
        super().__init__()
        self.attn=PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))
        self.ff=PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


if __name__ == "__main__": 
    model = attn_block().cuda()
    model.eval()
    inputs = torch.randn(1, 153, 1024).cuda()
    out = model(inputs)
    print(out.size())
    


# from poolformer import PatchEmbed, LayerNormChannel, GroupNorm, Mlp
# from typing import Sequence
# from functools import partial, reduce
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model
# def _cfg(url='', **kwargs):
#     return {
#         'url': url,
#         'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
#         'crop_pct': .95, 'interpolation': 'bicubic',
#         'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
#         'classifier': 'head',
#         **kwargs
#     }

# class AddPositionEmb(nn.Module):
#     """Module to add position embedding to input features
#     """
#     def __init__(
#         self, dim=384, spatial_shape=[14, 14],
#         ):
#         super().__init__()
#         if isinstance(spatial_shape, int):
#             spatial_shape = [spatial_shape]
#         assert isinstance(spatial_shape, Sequence), \
#             f'"spatial_shape" must by a sequence or int, ' \
#             f'get {type(spatial_shape)} instead.'
#         if len(spatial_shape) == 1:
#             embed_shape = list(spatial_shape) + [dim]
#         else:
#             embed_shape = [dim] + list(spatial_shape)
#         self.pos_embed = nn.Parameter(torch.zeros(1, *embed_shape))
#     def forward(self, x):
#         return x+self.pos_embed

# class Pooling(nn.Module):
#     """
#     Implementation of pooling for PoolFormer
#     --pool_size: pooling size
#     """
#     def __init__(self, pool_size=3, **kwargs):
#         super().__init__()
#         self.pool = nn.AvgPool2d(
#             pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

#     def forward(self, x):
#         return self.pool(x) - x

# class Attention(nn.Module):
#     """Attention module that can take tensor with [B, N, C] or [B, C, H, W] as input.
#     Modified from: 
#     https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
#     """
#     def __init__(self, dim, head_dim=32, qkv_bias=False, attn_drop=0., proj_drop=0.):
#         super().__init__()
#         assert dim % head_dim == 0, 'dim should be divisible by head_dim'
#         self.head_dim = head_dim
#         self.num_heads = dim // head_dim
#         self.scale = head_dim ** -0.5

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#     def forward(self, x):
#         shape = x.shape
#         if len(shape) == 4:
#             B, C, H, W = shape
#             N = H * W
#             x = torch.flatten(x, start_dim=2).transpose(-2, -1) # (B, N, C)
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

#         # trick here to make q@k.t more stable
#         attn = (q * self.scale) @ k.transpose(-2, -1)
#         # attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         if len(shape) == 4:
#             x = x.transpose(-2, -1).reshape(B, C, H, W)

#         return x

# class SpatialFc(nn.Module):
#     """SpatialFc module that take features with shape of (B,C,*) as input.
#     """
#     def __init__(
#         self, spatial_shape=[14, 14], **kwargs, 
#         ):
#         super().__init__()
#         if isinstance(spatial_shape, int):
#             spatial_shape = [spatial_shape]
#         assert isinstance(spatial_shape, Sequence), \
#             f'"spatial_shape" must by a sequence or int, ' \
#             f'get {type(spatial_shape)} instead.'
#         N = reduce(lambda x, y: x * y, spatial_shape)
#         self.fc = nn.Linear(N, N, bias=False)

#     def forward(self, x):
#         # input shape like [B, C, H, W]
#         shape = x.shape
#         x = torch.flatten(x, start_dim=2) # [B, C, H*W]
#         x = self.fc(x) # [B, C, H*W]
#         x = x.reshape(*shape) # [B, C, H, W]
#         return x

# class MetaFormerBlock(nn.Module):
#     """
#     Implementation of one MetaFormer block.
#     --dim: embedding dim
#     --token_mixer: token mixer module
#     --mlp_ratio: mlp expansion ratio
#     --act_layer: activation
#     --norm_layer: normalization
#     --drop: dropout rate
#     --drop path: Stochastic Depth, 
#         refer to https://arxiv.org/abs/1603.09382
#     --use_layer_scale, --layer_scale_init_value: LayerScale, 
#         refer to https://arxiv.org/abs/2103.17239
#     """
#     def __init__(self, dim, 
#                  token_mixer=nn.Identity, 
#                  mlp_ratio=4., 
#                  act_layer=nn.GELU, norm_layer=LayerNormChannel, 
#                  drop=0., drop_path=0., 
#                  use_layer_scale=True, layer_scale_init_value=1e-5):

#         super().__init__()

#         self.norm1 = norm_layer(dim)
#         self.token_mixer = token_mixer(dim=dim)
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
#                        act_layer=act_layer, drop=drop)
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.drop_path = DropPath(drop_path) if drop_path > 0. \
#             else nn.Identity()
#         self.use_layer_scale = use_layer_scale
#         if use_layer_scale:
#             self.layer_scale_1 = nn.Parameter(
#                 layer_scale_init_value * torch.ones((dim)), requires_grad=True)
#             self.layer_scale_2 = nn.Parameter(
#                 layer_scale_init_value * torch.ones((dim)), requires_grad=True)

#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


#     def forward(self, x):
#         if self.use_layer_scale:
#             x = x + self.drop_path(
#                 self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
#                 * self.token_mixer(self.norm1(x)))
#             x = x + self.drop_path(
#                 self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
#                 * self.mlp(self.norm2(x)))
#         else:
#             x = x + self.drop_path(self.token_mixer(self.norm1(x)))
#             x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x



# from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     print("3333",x.shape, window_size)
#     # [1, 4, 7, 4, 7, 295]
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows

# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image

#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x

# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#         pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
#     """

#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
#                  pretrained_window_size=[0, 0]):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.pretrained_window_size = pretrained_window_size
#         self.num_heads = num_heads

#         self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

#         # mlp to generate continuous relative position bias
#         self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
#                                      nn.ReLU(inplace=True),
#                                      nn.Linear(512, num_heads, bias=False))

#         # get relative_coords_table
#         relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
#         relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
#         relative_coords_table = torch.stack(
#             torch.meshgrid([relative_coords_h,
#                             relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
#         if pretrained_window_size[0] > 0:
#             relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
#             relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
#         else:
#             relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
#             relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
#         relative_coords_table *= 8  # normalize to -8, 8
#         relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
#             torch.abs(relative_coords_table) + 1.0) / np.log2(8)

#         self.register_buffer("relative_coords_table", relative_coords_table)

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=False)
#         if qkv_bias:
#             self.q_bias = nn.Parameter(torch.zeros(dim))
#             self.v_bias = nn.Parameter(torch.zeros(dim))
#         else:
#             self.q_bias = None
#             self.v_bias = None
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         qkv_bias = None
#         if self.q_bias is not None:
#             qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
#         print(qkv_bias.shape)
#         qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#         qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         # cosine attention
#         attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
#         logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
#         attn = attn * logit_scale

#         relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
#         relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, ' \
#                f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops

# class SwinTransformerBlock(nn.Module):
#     r""" Swin Transformer Block.

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#         pretrained_window_size (int): Window size in pre-training.
#     """

#     def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
#             pretrained_window_size=to_2tuple(pretrained_window_size))

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         if self.shift_size > 0:
#             # calculate attention mask for SW-MSA
#             H, W = self.input_resolution
#             img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#             h_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#             mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         else:
#             attn_mask = None

#         self.register_buffer("attn_mask", attn_mask)

#     def forward(self, x):
#         H, W = self.input_resolution
#         # print(x.shape)
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         x = x.view(B, H, W, C)

#         # cyclic shift
#         if self.shift_size > 0:
#             shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#         else:
#             shifted_x = x

#         # partition windows
#         x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
#         shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#         else:
#             x = shifted_x
#         x = x.view(B, H * W, C)
#         x = shortcut + self.drop_path(self.norm1(x))

#         # FFN
#         x = x + self.drop_path(self.norm2(self.mlp(x)))

#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.dim * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.window_size / self.window_size
#         flops += nW * self.attn.flops(self.window_size * self.window_size)
#         # mlp
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         # norm2
#         flops += self.dim * H * W
#         return flops

# try:
#     import os, sys

#     kernel_path = os.path.abspath(os.path.join('..'))
#     sys.path.append(kernel_path)
#     from kernels.window_process.window_process import WindowProcess, WindowProcessReverse

# except:
#     WindowProcess = None
#     WindowProcessReverse = None
#     print("[Warning] Fused window process have not been installed. Please refer to get_started.md for installation.")

# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x

# def window_partition(x, window_size):
#     """
#     Args:
#         x: (B, H, W, C)
#         window_size (int): window size

#     Returns:
#         windows: (num_windows*B, window_size, window_size, C)
#     """
#     B, H, W, C = x.shape
#     # 确保能整除
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows

# def window_reverse(windows, window_size, H, W):
#     """
#     Args:
#         windows: (num_windows*B, window_size, window_size, C)
#         window_size (int): Window size
#         H (int): Height of image
#         W (int): Width of image

#     Returns:
#         x: (B, H, W, C)
#     """
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x

# class WindowAttention(nn.Module):
#     r""" Window based multi-head self attention (W-MSA) module with relative position bias.
#     It supports both of shifted and non-shifted window.

#     Args:
#         dim (int): Number of input channels.
#         window_size (tuple[int]): The height and width of the window.
#         num_heads (int): Number of attention heads.
#         qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
#         attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
#         proj_drop (float, optional): Dropout ratio of output. Default: 0.0
#     """

#     def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

#         super().__init__()
#         self.dim = dim
#         self.window_size = window_size  # Wh, Ww
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         # define a parameter table of relative position bias
#         self.relative_position_bias_table = nn.Parameter(
#             torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

#         # get pair-wise relative position index for each token inside the window
#         coords_h = torch.arange(self.window_size[0])
#         coords_w = torch.arange(self.window_size[1])
#         coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#         coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#         relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#         relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#         relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
#         relative_coords[:, :, 1] += self.window_size[1] - 1
#         relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
#         relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#         self.register_buffer("relative_position_index", relative_position_index)

#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)

#         trunc_normal_(self.relative_position_bias_table, std=.02)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, mask=None):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         B_, N, C = x.shape
#         print(x.shape, ) # [16, 64, 295]
#         qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
#             self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
#         relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#         attn = attn + relative_position_bias.unsqueeze(0)

#         if mask is not None:
#             nW = mask.shape[0]
#             attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
#             attn = attn.view(-1, self.num_heads, N, N)
#             attn = self.softmax(attn)
#         else:
#             attn = self.softmax(attn)

#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x

#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

#     def flops(self, N):
#         # calculate flops for 1 window with token length of N
#         flops = 0
#         # qkv = self.qkv(x)
#         flops += N * self.dim * 3 * self.dim
#         # attn = (q @ k.transpose(-2, -1))
#         flops += self.num_heads * N * (self.dim // self.num_heads) * N
#         #  x = (attn @ v)
#         flops += self.num_heads * N * N * (self.dim // self.num_heads)
#         # x = self.proj(x)
#         flops += N * self.dim * self.dim
#         return flops

# class SwinTransformerBlock(nn.Module):
#     r""" Swin Transformer Block.

#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resulotion.
#         num_heads (int): Number of attention heads.
#         window_size (int): Window size.
#         shift_size (int): Shift size for SW-MSA.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
#         drop (float, optional): Dropout rate. Default: 0.0
#         attn_drop (float, optional): Attention dropout rate. Default: 0.0
#         drop_path (float, optional): Stochastic depth rate. Default: 0.0
#         act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
#         norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
#         fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
#     """

#     def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
#                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,
#                  fused_window_process=False):
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.num_heads = num_heads
#         self.window_size = window_size
#         self.shift_size = shift_size
#         self.mlp_ratio = mlp_ratio
#         if min(self.input_resolution) <= self.window_size:
#             # if window size is larger than input resolution, we don't partition windows
#             self.shift_size = 0
#             self.window_size = min(self.input_resolution)
#         assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

#         # self.norm1 = norm_layer(dim)
#         self.attn = WindowAttention(
#             dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
#             qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

#         if self.shift_size > 0:
#             # calculate attention mask for SW-MSA
#             H, W = self.input_resolution
#             img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
#             h_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             w_slices = (slice(0, -self.window_size),
#                         slice(-self.window_size, -self.shift_size),
#                         slice(-self.shift_size, None))
#             cnt = 0
#             for h in h_slices:
#                 for w in w_slices:
#                     img_mask[:, h, w, :] = cnt
#                     cnt += 1

#             mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
#             mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
#             attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
#             attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
#         else:
#             attn_mask = None

#         self.register_buffer("attn_mask", attn_mask)
#         self.fused_window_process = fused_window_process

#     def forward(self, x):
#         # 1, 1024, 295
#         H, W = self.input_resolution
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"

#         shortcut = x
#         print(self.dim)
#         # x = self.norm1(x)
#         x = x.view(B, H, W, C)

#         # cyclic shift
#         if self.shift_size > 0:
#             if not self.fused_window_process:
#                 shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
#                 # partition windows
#                 x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
#             else:
#                 x_windows = WindowProcess.apply(x, B, H, W, C, -self.shift_size, self.window_size)
#         else:
#             shifted_x = x
#             # partition windows
#             x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

#         x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

#         # W-MSA/SW-MSA
#         attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

#         # merge windows
#         attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

#         # reverse cyclic shift
#         if self.shift_size > 0:
#             if not self.fused_window_process:
#                 shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
#                 x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
#             else:
#                 x = WindowProcessReverse.apply(attn_windows, B, H, W, C, self.shift_size, self.window_size)
#         else:
#             shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
#             x = shifted_x
#         x = x.view(B, H * W, C)
#         x = shortcut + self.drop_path(x)
#         # FFN
#         x = x + self.drop_path(self.mlp(self.norm2(x)))
#         return x

#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
#                f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

#     def flops(self):
#         flops = 0
#         H, W = self.input_resolution
#         # norm1
#         flops += self.dim * H * W
#         # W-MSA/SW-MSA
#         nW = H * W / self.window_size / self.window_size
#         flops += nW * self.attn.flops(self.window_size * self.window_size)
#         # mlp
#         flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
#         # norm2
#         flops += self.dim * H * W
#         return flops



