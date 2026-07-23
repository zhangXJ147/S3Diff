import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, in_gate_channels, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        # self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_gate_channels, in_gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x, img_semantic=None):
        global channel_att_raw
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                if img_semantic is not None:
                    img_semantic_avg_pool = F.avg_pool2d(img_semantic.detach(), (img_semantic.size(2), img_semantic.size(3)),
                                                         stride=(img_semantic.size(2), img_semantic.size(3)))
                    avg_pool = torch.cat((avg_pool, img_semantic_avg_pool), dim=1)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                if img_semantic is not None:
                    img_semantic_max_pool = F.max_pool2d(img_semantic.detach(), (img_semantic.size(2), img_semantic.size(3)),
                                                         stride=(img_semantic.size(2), img_semantic.size(3)))
                    max_pool = torch.cat((max_pool, img_semantic_max_pool), dim=1)
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                if img_semantic is not None:
                    img_semantic_lp_pool = F.lp_pool2d(img_semantic.detach(), 2, (img_semantic.size(2), img_semantic.size(3)),
                                                       stride=(img_semantic.size(2), img_semantic.size(3)))
                    lp_pool = torch.cat((lp_pool, img_semantic_lp_pool), dim=1)
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                if img_semantic is not None:
                    img_semantic_lse_pool = logsumexp_2d(img_semantic.detach())
                    lse_pool = torch.cat((lse_pool, img_semantic_lse_pool), dim=1)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, use_semantic=False):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        if use_semantic:
            self.spatial = BasicConv(4, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
            # self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        else:
            self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x, img_semantic=None):
        x_compress = self.compress(x)
        if img_semantic is not None:
            img_semantic_compress = self.compress(img_semantic)
            x_compress = torch.cat((x_compress, img_semantic_compress), dim=1)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'],
                 no_spatial=False, use_concat=False, use_semantic=False):
        super(CBAM, self).__init__()
        if use_semantic:
            in_gate_channels = gate_channels * 2
        else:
            in_gate_channels = gate_channels
        self.ChannelGate = ChannelGate(in_gate_channels, gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        self.use_concat = use_concat
        if not no_spatial:
            self.SpatialGate = SpatialGate(use_semantic=use_semantic)
        if use_concat:
            self.conv = BasicConv(2 * gate_channels, gate_channels, 3, padding=1)

    def forward(self, x, img_semantic=None):
        x_out = self.ChannelGate(x, img_semantic=img_semantic)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out, img_semantic=img_semantic)
        if self.use_concat:
            x_out = torch.cat((x_out, x), dim=1)
            x_out = self.conv(x_out)
        return x_out
