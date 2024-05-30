from __future__ import annotations 
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
from torch.nn import functional as F 
from collections.abc import Sequence
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
from monai.networks.nets.segresnet import SegResNet,SegResNetVAE
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
            nn.ReLU(),
            nn.Conv3d(32,16,3,3,2),
            nn.GroupNorm(16,16,eps=0.00001,affine=True),
            nn.ReLU(), 
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
        return super().forward(x)
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
class SegResVAE(SegResNetVAE):
    def __init__(self, input_image_size: Sequence[int], vae_estimate_std: bool = False, vae_default_std: float = 0.3, vae_nz: int = 256, spatial_dims: int = 3, init_filters: int = 8, 
    in_channels: int = 1, out_channels: int = 2, dropout_prob: Optional[float] = None, 
    act: Union[str, tuple] = ("RELU",{"inplace":True}),
    norm: Union[Tuple, str] = ("GROUP",{"num_groups":8}),
    use_conv_final: bool = True, blocks_down: tuple = (1,2,2,4), 
    blocks_up: tuple = (1,1,1), upsample_mode: Union[UpsampleMode, str] = UpsampleMode.NONTRAINABLE):
        super().__init__(input_image_size, vae_estimate_std, vae_default_std, vae_nz, spatial_dims, init_filters, in_channels, out_channels, dropout_prob, act, norm, use_conv_final, blocks_down, blocks_up, upsample_mode)
    def forward(self, x):
        out,loss = super().forward(x)
        if self.training: 
            return out,loss 
        else: 
            return out
    def get_recon(self,x): 
        vae_input,_ = self.encode(x)
        x_vae = self.vae_down(vae_input)
        x_vae = x_vae.view(-1, self.vae_fc1.in_features)
        z_mean = self.vae_fc1(x_vae)

        z_mean_rand = torch.randn_like(z_mean)
        z_mean_rand.requires_grad_(False)
        if self.vae_estimate_std:
            z_sigma = self.vae_fc2(x_vae)
            z_sigma = F.softplus(z_sigma)
            vae_reg_loss = 0.5 * torch.mean(z_mean**2 + z_sigma**2 - torch.log(1e-8 + z_sigma**2) - 1)

            x_vae = z_mean + z_sigma * z_mean_rand
        else:
            z_sigma = self.vae_default_std
            vae_reg_loss = torch.mean(z_mean**2)

            x_vae = z_mean + z_sigma * z_mean_rand
        x_vae = self.vae_fc3(x_vae)
        x_vae = self.act_mod(x_vae)
        x_vae = x_vae.view([-1, self.smallest_filters] + self.fc_insize)
        x_vae = self.vae_fc_up_sample(x_vae)

        for up, upl in zip(self.up_samples, self.up_layers):
            x_vae = up(x_vae)
            x_vae = upl(x_vae)

        x_vae = self.vae_conv_final(x_vae)
        return x_vae 


class DinsdaleDomainPred(nn.Module):
    def __init__(self, n_domains,init_features=4,num_blocks=4) -> None:
        super().__init__()
        block_list = list() 
        feats = init_features
        for i in range(num_blocks):
            module = nn.Sequential(DinsdaleHalfBlock(feats,feats),
            nn.MaxPool3d(kernel_size=2,stride=2)) 
            block_list.append(module)
        block_list.append(nn.Flatten(1))
        block_list.append(nn.Linear(512,256))
        block_list.append(nn.ReLU())
        self.decoder_block = nn.Sequential(*block_list)

        self.projector = nn.Sequential(DinsdaleProjector(init_features*64,1),
        nn.Flatten(1),nn.Linear(in_features=5832,out_features=256),nn.ReLU())
        self.classi = nn.Sequential(nn.Linear(512,124),nn.ReLU(),nn.Linear(124,2))
    def forward(self,x): 
        (bottleneck,feat_map)= x 
        flat_feat_map = self.decoder_block(feat_map)
        bottle_feats =  self.projector(bottleneck)
        combi = torch.cat([bottle_feats,flat_feat_map],dim=1)
        return self.classi(combi)


class DinsdaleProjector(nn.Module):
    def __init__(self, in_feats,out_feats) -> None:
        super().__init__() 
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_feats,out_channels=out_feats,kernel_size=1,padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(out_feats)
        )
    def forward(self,x):
        return self.block(x)



class DinsdaleHalfBlock(nn.Module):
    def __init__(self,in_feats,out_feats) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels=in_feats,out_channels=out_feats,kernel_size=3,padding=1,bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(out_feats)
        )
    def forward(self,x):
        return self.block(x)
class SegResneDinsdale(SegResNet):
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
        self.discrim = DinsdaleDomainPred(n_domains=2)
    def forward(self,x): 
        x, down_x = self.encode(x)
        down_x.reverse()
        x = self.decode(x, down_x)
        domain_input = torch.cat([down_x[0],x],dim=1)
        domain_pred = self.discrim(domain_input)
        if self.training or self.infer_phase: 
            return x,domain_pred 
        else: 
            return x 
    def set_infer_phase(self,infer_phase=False): 
        self.infer_phase=infer_phase 
