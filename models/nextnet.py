import math
import os

import numpy
import torch
from torch import nn
from models.modules import ConvNextBlock, SinusoidalPosEmb
from models.convnextv2 import ConvNeXtV2Block
from models.performer_pytorch import Performer
from einops import rearrange


def exists(x):
    return x is not None


class NextNet(nn.Module):
    """
    A backbone model comprised of a chain of ConvNext blocks, with skip connections.
    The skip connections are connected similar to a "U-Net" structure (first to last, middle to middle, etc).
    """

    def __init__(self, in_channels=3, out_channels=3, depth=16, filters_per_layer=64, frame_conditioned=False,
                 use_cbam=False, use_concat=False):
        """
        Args:
            in_channels (int):
                Number of input image channels.
            out_channels (int):
                Number of network output channels.
            depth (int):
                Number of ConvNext blocks in the network.
            filters_per_layer (int):
                Base dimension in each ConvNext block.
            frame_conditioned (bool):
                Whether to condition the network on the difference between the current and previous frames. Should
                be True when training a DDPM frame predictor.
        """
        super().__init__()

        if isinstance(filters_per_layer, (list, tuple)):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        time_dim = dims[0]
        emb_dim = time_dim * 2 if frame_conditioned else time_dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # First block doesn't have a normalization layer
        self.layers.append(ConvNextBlock(in_channels, dims[0], emb_dim=emb_dim, norm=False,
                                         ))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNextBlock(dims[i - 1], dims[i], emb_dim=emb_dim, norm=True,
                                             ))
        self.layers.append(
            ConvNextBlock(2 * dims[math.ceil(self.depth / 2) - 1], dims[math.ceil(self.depth / 2)], emb_dim=emb_dim,
                          norm=True,
                          use_cbam=use_cbam, use_concat=use_concat,
                          ))
        for i in range(math.ceil(self.depth / 2) + 1, depth - 1):
            self.layers.append(ConvNextBlock(2 * dims[i - 1], dims[i], emb_dim=emb_dim, norm=True,
                                             use_cbam=use_cbam, use_concat=use_concat,
                                             ))
        for i in range(depth - 1, depth):
            self.layers.append(ConvNextBlock(2 * dims[i - 1], dims[i], emb_dim=emb_dim, norm=True,
                                             use_cbam=use_cbam, use_concat=use_concat,
                                             ))

        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.Conv2d(dims[depth - 1], out_channels, 1)

        # Encoder for positional embedding of timestep
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        if frame_conditioned:
            # Encoder for positional embedding of frame
            self.frame_encoder = nn.Sequential(
                SinusoidalPosEmb(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.GELU(),
                nn.Linear(time_dim * 4, time_dim)
            )

    def forward(self, x, t, frame_diff=None, img_semantic=None):
        time_embedding = self.time_encoder(t)

        if frame_diff is not None:
            frame_embedding = self.frame_encoder(frame_diff)
            embedding = torch.cat([time_embedding, frame_embedding], dim=1)
        else:
            embedding = time_embedding

        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, embedding, img_semantic=img_semantic)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, embedding, img_semantic=img_semantic)

        return self.final_conv(x)


class ModifiedNextNet(NextNet):
    def __init__(self, in_channels=3, out_channels=3, depth=16, filters_per_layer=64, frame_conditioned=False,
                 use_cbam=False, use_concat=False, use_semantic=False,
                 semantic_dim=512, nheads=1, dhead=64,
                 ):
        super(ModifiedNextNet, self).__init__(in_channels=in_channels, out_channels=out_channels, depth=depth,
                                              filters_per_layer=filters_per_layer, frame_conditioned=frame_conditioned,
                                              use_cbam=use_cbam, use_concat=use_concat)

        self.use_semantic = use_semantic
        if use_semantic:
            self.att = Performer(dim=filters_per_layer, heads=nheads, dim_head=dhead, context_dim=semantic_dim)
            self.conv = nn.utils.spectral_norm(nn.Conv2d(filters_per_layer + filters_per_layer, filters_per_layer, 3, padding=1))
            self.act = nn.GELU()

    def forward(self, x, t, frame_diff=None, img_semantic=None):
        time_embedding = self.time_encoder(t)

        if frame_diff is not None:
            frame_embedding = self.frame_encoder(frame_diff)
            embedding = torch.cat([time_embedding, frame_embedding], dim=1)
        else:
            embedding = time_embedding

        residuals = []

        i = 0
        j = 8
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, embedding)
            residuals.append(x)
            i += 1
            if self.use_semantic and i == j:
                curr_h, curr_w = x.shape[-2], x.shape[-1]
                img_semantic = nn.functional.interpolate(img_semantic.detach(), (curr_h, curr_w), mode="bicubic")
                h_pfm = self.att(x, img_semantic)
                x = self.act(self.conv(torch.cat((x, h_pfm), dim=1)))
        # x_npy = rearrange(x[0], 'c h w -> (h w) c').contiguous().cpu().numpy()
        # t_str = str(t.cpu()[0].item())
        # dir_path = './tsne/s3diff_imagenet50_42/'
        # os.makedirs(dir_path, exist_ok=True)
        # numpy.save(f'{dir_path}/{t_str}.npy', x_npy)
        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, embedding)
            i += 1
            if self.use_semantic and i == j:
                curr_h, curr_w = x.shape[-2], x.shape[-1]
                img_semantic = nn.functional.interpolate(img_semantic.detach(), (curr_h, curr_w), mode="bicubic")
                h_pfm = self.att(x, img_semantic)
                x = self.act(self.conv(torch.cat((x, h_pfm), dim=1)))

        return self.final_conv(x)


class NextNetV2(nn.Module):
    """
    A backbone model comprised of a chain of ConvNextV2 blocks, with skip connections.
    The skip connections are connected similar to a "U-Net" structure (first to last, middle to middle, etc).
    """

    def __init__(self, in_channels=3, out_channels=3, depth=16, filters_per_layer=64, frame_conditioned=False,
                 use_cbam=False, use_concat=False, use_semantic=False,
                 ):
        """
        Args:
            in_channels (int):
                Number of input image channels.
            out_channels (int):
                Number of network output channels.
            depth (int):
                Number of ConvNext blocks in the network.
            filters_per_layer (int):
                Base dimension in each ConvNext block.
            frame_conditioned (bool):
                Whether to condition the network on the difference between the current and previous frames. Should
                be True when training a DDPM frame predictor.
        """
        super().__init__()

        if isinstance(filters_per_layer, (list, tuple)):
            dims = filters_per_layer
        else:
            dims = [filters_per_layer] * depth

        time_dim = dims[0]
        emb_dim = time_dim * 2 if frame_conditioned else time_dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        # First block doesn't have a normalization layer
        self.layers.append(ConvNeXtV2Block(in_channels, dims[0], emb_dim=emb_dim,
                                           use_semantic=use_semantic))

        for i in range(1, math.ceil(self.depth / 2)):
            self.layers.append(ConvNeXtV2Block(dims[i - 1], dims[i], emb_dim=emb_dim,
                                               use_semantic=use_semantic
                                               ))
        for i in range(math.ceil(self.depth / 2), depth):
            self.layers.append(ConvNeXtV2Block(2 * dims[i - 1], dims[i], emb_dim=emb_dim,
                                               use_cbam=use_cbam, use_concat=use_concat,
                                               # use_semantic=use_semantic
                                               ))

        # After all blocks, do a 1x1 conv to get the required amount of output channels
        self.final_conv = nn.Conv2d(dims[depth - 1], out_channels, 1)

        # Encoder for positional embedding of timestep
        self.time_encoder = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim)
        )

        if frame_conditioned:
            # Encoder for positional embedding of frame
            self.frame_encoder = nn.Sequential(
                SinusoidalPosEmb(time_dim),
                nn.Linear(time_dim, time_dim * 4),
                nn.GELU(),
                nn.Linear(time_dim * 4, time_dim)
            )

    def forward(self, x, t, frame_diff=None, img_semantic=None):
        time_embedding = self.time_encoder(t)

        if frame_diff is not None:
            frame_embedding = self.frame_encoder(frame_diff)
            embedding = torch.cat([time_embedding, frame_embedding], dim=1)
        else:
            embedding = time_embedding

        residuals = []
        for layer in self.layers[0: math.ceil(self.depth / 2)]:
            x = layer(x, embedding, img_semantic=img_semantic)
            residuals.append(x)

        for layer in self.layers[math.ceil(self.depth / 2): self.depth]:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = layer(x, embedding, img_semantic=img_semantic)

        return self.final_conv(x)
