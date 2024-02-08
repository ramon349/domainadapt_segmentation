from monai.networks.nets.unet import Unet
from monai.networks.nets.dynunet import DynUNet
import warnings
from typing import Optional, Sequence, Tuple, Union
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
