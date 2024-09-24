
from __future__ import annotations 
from monai.networks.nets.segresnet import SegResNet 
import warnings
from typing import Optional, Sequence, Tuple, Union
from monai.utils import UpsampleMode
from torch import nn
import torch 
import pdb 
import torch
from torch.autograd import Function
import time 
from torch.nn import functional as F 
from collections.abc import Sequence
from .model_factory import ModelRegister 

@ModelRegister.register(cls_name="3DSegResProto")
class SegResNetProto(SegResNet):
    def __init__(self,         spatial_dims: int = 3,
        init_filters: int = 8,
        in_channels: int = 1,
        out_channels: int = 2,
        dropout_prob: float | None = None,
        act: tuple | str = ("RELU", {"inplace": True}),
        norm: tuple | str = ("GROUP", {"num_groups": 8}),
        norm_name: str = "",
        num_groups: int = 8,
        use_conv_final: bool = True,
        blocks_down: tuple = (1, 2, 2, 4),
        blocks_up: tuple = (1, 1, 1),
        upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE,
    ):
        super().__init__(spatial_dims=spatial_dims, init_filters=init_filters, in_channels=in_channels, out_channels=out_channels, dropout_prob=dropout_prob, act=act, norm=norm, norm_name=norm_name, num_groups=num_groups, use_conv_final=use_conv_final, blocks_down=blocks_down, blocks_up=blocks_up, upsample_mode=upsample_mode)

    def forward(self,x,only_feature=False): 
        x, down_x = self.encode(x)
        down_x.reverse()
        feats,seg_out = self.decode(x, down_x)
        if self.training  or only_feature :
            return feats ,seg_out
        else: 
            return  seg_out
    def decode(self, x, down_x):
        for i, (up, upl) in enumerate(zip(self.up_samples, self.up_layers)):
            x = up(x) + down_x[i + 1]
            x = upl(x)

        if self.use_conv_final:
            seg_pair= self.conv_final(x)

        return x,seg_pair