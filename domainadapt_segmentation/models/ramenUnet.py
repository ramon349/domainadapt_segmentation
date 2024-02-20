from monai.networks.nets.unet import Unet
from monai.networks.nets.dynunet import DynUNet
import warnings
from typing import Optional, Sequence, Tuple, Union
from monai.utils import UpsampleMode
from torch import nn
import torch 
import pdb 
import torch
from torch.autograd import Function
import time 

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
class linearModel(nn.Module):
    def __init__(self,in_feats) -> None:
        super().__init__() 
        self.model = nn.Linear(in_feats,2) 
    def forward(self,x): 
        x = x.view(x.size(0),-1)
        print(x.shape)
        return self.model(x)
#TODO: can i adapt so a regular unet does this lol no 
class DynUnetSingleBias(DynUNet):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Sequence[Union[Sequence[int], int]], strides: Sequence[Union[Sequence[int], int]], upsample_kernel_size: Sequence[Union[Sequence[int], int]], filters: Optional[Sequence[int]] = None, dropout: Optional[Union[Tuple, str, float]] = None, norm_name: Union[Tuple, str] = 'INSTANCE' ,act_name: Union[Tuple, str] = 'LEAKYRELU', deep_supervision: bool = False, deep_supr_num: int = 1, res_block: bool = False, trans_bias: bool = False,config=None):
        super().__init__(spatial_dims, in_channels, out_channels, kernel_size, strides, upsample_kernel_size, filters, dropout, norm_name, act_name, deep_supervision, deep_supr_num, res_block, trans_bias)
        samples = config['batch_size'] * config['rand_crop_label_num_samples']
        self.input_dim = torch.tensor(config['spacing_vox_dim']) 
        self.num_convs  =2 
        self.num_feats = 8 
        self.num_in_features = self.figure_cals(self.num_feats)
        self.lin_layer = nn.Sequential(
            nn.Conv3d(32,16,kernel_size=3,stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16,8,kernel_size=3,stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(8),
            linearModel(in_feats=  self.num_in_features)
        )
    def forward(self, x):
        mid = self.skip_layers(x)
        print(f"Mid shape is {mid.shape}")
        demo_pred = self.lin_layer(mid) 
        out = self.output_block(mid) 
        if self.training: 
            return out,demo_pred
        else: 
            return out
    def figure_cals(self,num_channels): 
        c_dim = self.input_dim 
        #these are just the defaults 

        dilations = torch.tensor([1,1,1]) 
        paddings = torch.tensor([0,0,0])
        kernels = torch.tensor([3,3,3])
        stride= torch.tensor([2,2,2])
        for e in range(self.num_convs): 
            c_dim = calc_output_shape(c_dim,padding=paddings,dilation=dilations,kernel_size=kernels,stride=stride)
        return  int(torch.prod(c_dim).item()*num_channels)
class  DynUnetMultiBias(DynUNet): 
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Sequence[Union[Sequence[int], int]], strides: Sequence[Union[Sequence[int], int]], upsample_kernel_size: Sequence[Union[Sequence[int], int]], filters: Optional[Sequence[int]] = None, dropout: Optional[Union[Tuple, str, float]] = None, norm_name: Union[Tuple, str] = 'INSTANCE' ,act_name: Union[Tuple, str] = 'LEAKYRELU', deep_supervision: bool = False, deep_supr_num: int = 1, res_block: bool = False, trans_bias: bool = False,config=None):
        super().__init__(spatial_dims, in_channels, out_channels, kernel_size, strides, upsample_kernel_size, filters, dropout, norm_name, act_name, deep_supervision, deep_supr_num, res_block, trans_bias)
        samples = config['batch_size'] * config['rand_crop_label_num_samples']
        self.input_dim = torch.tensor(config['spacing_vox_dim']) 
        self.num_convs  =2 
        self.num_feats = 8 
        self.num_in_features = self.figure_cals()
        self.lin_layer = nn.Sequential(
            nn.Conv3d(32,16,kernel_size=3,stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16,8,kernel_size=3,stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(8),
            linearModel(in_feats=  self.num_in_features)
        )
        self.mask_layer =  nn.Sequential(
            nn.Conv3d(2,1,kernel_size=3,stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(1,1,kernel_size=3,stride=2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(8),
            linearModel(in_feats=  self.num_in_features)
        )
    def forward(self, x):
        mid = self.skip_layers(x)
        print(f"Mid shape is {mid.shape}")
        demo_pred = self.lin_layer(mid) 
        out = self.output_block(mid) 
        mask_demo_pred = out()
        if self.training: 
            return out,demo_pred
        else: 
            return out
    def figure_cals(self): 
        c_dim = self.input_dim 
        #these are just the defaults 
        dilations = torch.tensor([1,1,1]) 
        paddings = torch.tensor([0,0,0])
        kernels = torch.tensor([3,3,3])
        stride= torch.tensor([2,2,2])
        for e in range(self.num_convs): 
            c_dim = calc_output_shape(c_dim,padding=paddings,dilation=dilations,kernel_size=kernels,stride=stride)
        return  int(torch.prod(c_dim).item()*self.num_feats)


def calc_output_shape(arr_dim,padding=0,dilation=0,kernel_size=0,stride=0):
    numerator = arr_dim + 2*padding - dilation*(kernel_size-1)-1 
    denum = stride
    return torch.floor((numerator/denum) + 1 ) 
from monai.networks.nets.segresnet import SegResNet
class segResnetBias(SegResNet):
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
        self.comp = nn.Sequential(
            nn.Conv3d(64,32,3,3,2),
            nn.GroupNorm(32,32,eps=0.00001,affine=True),
            nn.Conv3d(32,16,3,3,2),
            nn.GroupNorm(16,16,eps=0.00001,affine=True)
        )
    def forward(self,x): 
        x, down_x = self.encode(x)
        down_x.reverse()
        embed = self.comp(down_x[0] )
        flat_vec = embed.flatten(1)
        x = self.decode(x, down_x)
        if self.training: 
            return x, flat_vec
        else: 
            return x 
        