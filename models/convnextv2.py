# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from einops import rearrange
from models.cbam import CBAM
from torch.nn.utils import spectral_norm
from models.module_attention import ModifiedSpatialTransformer
from models.attention import SpatialTransformer
from models.performer_pytorch import Performer
from models.polaformer import PolaFormer


def exists(x):
    return x is not None


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=4, drop_path=0.,
                 semantic_dim=512, nheads=1, dhead=64,
                 use_cbam=False, use_concat=False,
                 use_semantic=False,
                 ):
        super().__init__()
        self.use_semantic = use_semantic
        self.use_cbam = use_cbam

        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(emb_dim, dim)
        ) if exists(emb_dim) else None

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mult * dim_out)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.silu = nn.SiLU()
        self.grn = GRN(mult * dim_out)
        self.pwconv2 = nn.Linear(mult * dim_out, dim_out)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        if use_cbam:
            self.cbam = CBAM(dim_out, 16, use_concat=use_concat)
        else:
            self.cbam = None

        if use_semantic:
            # self.att = ModifiedSpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=dim_out)
            # self.att = SpatialTransformer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=context_dim)
            self.att = Performer(dim=semantic_dim, heads=nheads, dim_head=dhead, context_dim=dim_out)
            # self.att = PolaFormer(in_channels=semantic_dim, n_heads=nheads, d_head=dhead, context_dim=dim_out)
            self.conv = spectral_norm(nn.Conv2d(dim_out + semantic_dim, dim_out, 3, padding=1))
            self.act = nn.GELU()

    def forward(self, x, emb=None, img_semantic=None):
        input = x
        x = self.dwconv(x)

        if exists(self.mlp):
            assert exists(emb), 'time (and possibly frame) emb must be passed in'
            condition = self.mlp(emb)
            x = x + rearrange(condition, 'b c -> b c 1 1')

        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.act(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.silu(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.use_semantic and img_semantic is not None:
            curr_h, curr_w = x.shape[-2], x.shape[-1]
            img_semantic = F.interpolate(img_semantic.detach(), (curr_h, curr_w), mode="bicubic")
            h_pfm = self.att(img_semantic, x)
            x = self.act(self.conv(torch.cat((x, h_pfm), dim=1)))

        if self.use_cbam:
            assert self.cbam is not None
            x = self.cbam(x)

        x = self.res_conv(input) + self.drop_path(x)
        return x
