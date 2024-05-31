from __future__ import annotations 
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
from monai.networks.nets.segresnet import SegResNet


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        out = grad_output.neg()
        return out 

def grad_reverse(x):
    return GradReverse.apply(x) 

ModelRegister.add_model("3DSegRes",SegResNet)    
@ModelRegister.register("3DSegResOneBias")
class SegResnetBiasClassiOneBranch(SegResNet):
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
        self.bottleneck_branch= nn.Sequential(
            nn.Conv3d(64,32,3,3,2),
            nn.GroupNorm(32,32,eps=0.00001,affine=True),
            nn.ReLU(),
            nn.Conv3d(32,16,3,3,2),
            nn.ReLU(),
            nn.GroupNorm(16,16,eps=0.00001,affine=True),
            nn.Flatten(),
            nn.Linear(432,432),
            nn.ReLU(),
            nn.Linear(432,2),
        )
        self.infer_phase=False
    def forward(self,x): 
        x, down_x = self.encode(x)
        down_x.reverse()
        embed = self.bottleneck_branch(down_x[0] )
        x = self.decode(x, down_x)
        if self.training or self.infer_phase: 
            return x, embed
        else: 
            return x 
    def set_infer_phase(self,status=False):
        self.infer_phase=status
@ModelRegister.register("3DSegResOneBiasAdv")
class SegResnetBiasClassiOneBranchAdv(SegResnetBiasClassiOneBranch):
    def __init__(self, spatial_dims: int = 3, init_filters: int = 8, in_channels: int = 1, out_channels: int = 2, dropout_prob: float | None = None, act: tuple | str = ("RELU", { "inplace": True }), norm: tuple | str = ("GROUP", { "num_groups": 8 }), norm_name: str = "", num_groups: int = 8, use_conv_final: bool = True, blocks_down: tuple = (1, 2, 2, 4), blocks_up: tuple = (1, 1, 1), upsample_mode: UpsampleMode | str = UpsampleMode.NONTRAINABLE):
        super().__init__(spatial_dims, init_filters, in_channels, out_channels, dropout_prob, act, norm, norm_name, num_groups, use_conv_final, blocks_down, blocks_up, upsample_mode)
        self.debias = False
    def forward(self, x): 
        x, down_x = self.encode(x)
        down_x.reverse()
        if self.training and self.debias:
            embed = self.bottleneck_branch( grad_reverse(down_x[0] ))
        else: 
            embed = self.bottleneck_branch( grad_reverse(down_x[0] ))
        x = self.decode(x, down_x)
        if self.training or self.infer_phase: 
            return x, embed
        else: 
            return x 
@ModelRegister.register("3DSegResTwoBias")
class SegResnetBiasClassiTwoBranch(SegResNet):
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
        self.bottleneck_branch = nn.Sequential(
            nn.Conv3d(64,32,3,3,2),
            nn.GroupNorm(32,32,eps=0.00001,affine=True),
            nn.ReLU(),
            nn.Conv3d(32,16,3,3,2),
            nn.ReLU(),
            nn.GroupNorm(16,16,eps=0.00001,affine=True),
            nn.Flatten(),
            nn.Linear(432,432),
            nn.ReLU(),
            nn.Linear(432,2),
        )
        self.mask_branch = nn.Sequential(
            nn.Conv3d(2,4,3,3,2),
            nn.GroupNorm(4,4,eps=0.00001,affine=True),
            nn.ReLU(),
            nn.Conv3d(4,8,3,3,2),
            nn.ReLU(),
            nn.Conv3d(8,10,3,3,2),
            nn.GroupNorm(10,10,eps=0.00001,affine=True),
            nn.Conv3d(10,12,3,3,2),
            nn.Flatten(),
            nn.Linear(324,324),
            nn.ReLU(),
            nn.Linear(324,2)
        )
        self.infer_phase=False
    def forward(self,x): 
        x, down_x = self.encode(x)
        down_x.reverse()
        block_embed = self.bottleneck_branch(down_x[0])
        x = self.decode(x, down_x)
        mask_embed = self.mask_branch(x)
        if self.training or self.infer_phase: 
            return x, block_embed,mask_embed
        else: 
            return x 
    def set_infer_phase(self,infer_phase=False): 
        self.infer_phase=infer_phase 
