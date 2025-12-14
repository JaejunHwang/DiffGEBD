import itertools
import math

import einops
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torchvision.ops.misc import FrozenBatchNorm2d
from .resnet import ResNetXX
from .diff_former import DiffFormer


class CrossSE_1d(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CrossSE_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        b, c, _ = x.size()
        x = self.avg_pool(x).view(b, c)
        x = self.fc(x).view(b, c, 1)
        return y * x.expand_as(y)

class DiffHead(nn.Module):
    def __init__(self, dim, num_layers, num_blocks, num_windows, group=1, resnet_type=1, similarity_func='cosine', offset=0):
        super(DiffHead, self).__init__()
        self.dim = dim
        self.out_channels = dim * 1
        self.group = group
        self.similarity_func = similarity_func
        self.offset = offset
        self.nl = num_layers
        self.nb = num_blocks
        self.nw = num_windows
    
        self.t_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.ReLU()) for _ in range(self.nl)])
        self.ddm_encoder = ResNetXX(self.nl * group, dim, resnet_type=resnet_type)


    def forward(self, x_fps):

        out_list = []
        for i, xi in enumerate(x_fps):
            t_feat = 0.5 * (self.t_conv[i](xi) + xi) # point-wise conv
            out_list.extend([t_feat])
        out_list = torch.stack(out_list, dim=1)  # (b*nw, nl, c, w)

        diffs = out_list

        x = diffs.permute(0, 3, 1, 2).contiguous()  # (b*nw, w, nl*2, c)
        
        sim = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)  # (b*nw, w, w, nl)
        
        diff_map = sim.permute(0, 3, 1, 2).contiguous()  # (b*nw, nl, w, w)
        diff_map = self.ddm_encoder(diff_map).mean(-1).mean(-1) # (b*nw, c)
        
        h = diff_map
        h = einops.rearrange(h, "(b nw) c -> b c nw", nw=self.nw)

        return h

class DiffHead_former(nn.Module):
    def __init__(self, dim, num_layers, num_blocks, num_windows, group=1, resnet_type=1, similarity_func='cosine', 
                 offset=0, diff_idx=0):
        super(DiffHead_former, self).__init__()
        self.dim = dim
        self.out_channels = dim * 1
        self.group = group
        self.similarity_func = similarity_func
        self.offset = offset
        self.nl = num_layers
        self.nb = num_blocks
        self.nw = num_windows
        self.diff_idx = diff_idx

        self.t_conv = nn.ModuleList([nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim), nn.ReLU()) for _ in range(self.nl)])

        self.diff_former = DiffFormer(dim, diff_idx=self.diff_idx)
        self.ddm_encoder = ResNetXX(self.nl * group * num_blocks, dim, resnet_type=resnet_type)
        self.out_conv = nn.Sequential(
            nn.Conv1d(dim * self.nl, dim, kernel_size=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, 1)
        )
        self.crossse1_1d = CrossSE_1d(dim)
        self.crossse2_1d = CrossSE_1d(dim)
        self.fuse = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=True), nn.ReLU(inplace=True),
        )


    def forward(self, x_fps):

        out_list = []
        for i, xi in enumerate(x_fps):
            t_feat = 0.5 * (self.t_conv[i](xi) + xi) # point-wise conv
            out_list.extend([t_feat])
        out_list = torch.stack(out_list, dim=1)  # (b*nw, nl, c, w)
        
        x = einops.rearrange(out_list, "(b nw) nl c nf -> (b nl nw) c nf", nw=self.nw, nl=self.nl)
        out, out_diff_lists = self.diff_former(x)
        diffs = []
        nf = out_diff_lists[0].shape[-1]
        for diff in out_diff_lists:
            nf_ = diff.shape[-1]
            if nf_ < nf:
                diff = torch.nn.functional.interpolate(diff, scale_factor=nf/nf_, mode='area')
            diff = einops.rearrange(diff, "(b nl nw) c nf -> (b nw) nl c nf", nw=self.nw, nl=self.nl)
            diffs.append(diff)
        # (b*nw, nl*3, nf, c)
        diffs = torch.cat(diffs, dim=1).permute(0, 1, 3, 2).contiguous()
        x = diffs.permute(0, 2, 1, 3).contiguous()  # (b*nw, nf, nl*3, c)

        sim = F.cosine_similarity(x.unsqueeze(2), x.unsqueeze(1), dim=-1)  # (b*nw, nf, nf, nl*3)
        diff_map = sim.permute(0, 3, 1, 2).contiguous()  # (b*nw, nl*3, nf, nf)
        diff_map = self.ddm_encoder(diff_map)
        # (b*nw, 512, nf_down_down)
        diff_map = torch.mean(diff_map, dim=-1)

        out = einops.rearrange(out, "(b nl) c nf -> b (c nl) nf", nl=self.nl)
        # (b*nw, 512, 3)
        out = self.out_conv(out)

        img2ddm = self.crossse1_1d(out, diff_map).mean(-1)
        ddm2img = self.crossse2_1d(diff_map, out).mean(-1)

        # (b*nw, 1024)
        h = torch.cat([img2ddm, ddm2img], dim=1)
        # (b, 512, nw)
        h = self.fuse(h).reshape(-1, self.dim, self.nw)
        return h
             

class BasicGEBD(Module):
    def __init__(self, cfg):
        super().__init__()
        
        dim = cfg.MODEL.DIMENSION
        window_size = cfg.MODEL.WINDOW_SIZE
        seq_len = cfg.INPUT.SEQUENCE_LENGTH
        self.num_layers = cfg.MODEL.HEAD_CHOICE-cfg.MODEL.FPN_START_IDX+1
        self.k = window_size // 2
        self.window_size = window_size
        self.diff_head = DiffHead_former(dim, num_blocks=cfg.MODEL.NUM_BLOCKS, num_layers=self.num_layers, resnet_type=1, num_windows=seq_len)

    def forward(self, x):
        """
        Args:
            x: (b c t)
        """
        B, nl, C, T = x.shape

        x_fps = einops.rearrange(x, "b nl c t -> (b nl) c t", b=B, c=C) # nl : number of layers
        x_fps = F.pad(x_fps, pad=(self.k, self.k), mode='replicate').unsqueeze(2)  # (b*nl, c, t+2k)
        x_fps = F.unfold(x_fps, kernel_size=(1, self.window_size)) # window_size = 2k+1, # (b*nl c*w t) w: window size
        x_fps = einops.rearrange(x_fps, "(b nl) (c nf) t -> nl (b t) c nf", b=B, c=C) # Head Batch Tem Ch Win

        feat = self.diff_head(x_fps) # b c t
        
        return feat
    