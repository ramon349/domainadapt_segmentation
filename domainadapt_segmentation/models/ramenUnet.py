from monai.networks.nets.unet import Unet
from monai.networks.nets.dynunet import DynUNet
import warnings
from typing import Optional, Sequence, Tuple, Union
from torch import nn

class linearModel(nn.Module):
    def __init__(self) -> None:
        super().__init__() 
        self.model = nn.Linear(36844,2) 
    def forward(self,x): 
        x = x.view(x.size(0),-1)
        return self.model(x)
class RamenUnet(DynUNet):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Sequence[Union[Sequence[int], int]], strides: Sequence[Union[Sequence[int], int]], upsample_kernel_size: Sequence[Union[Sequence[int], int]], filters: Optional[Sequence[int]] = None, dropout: Optional[Union[Tuple, str, float]] = None, norm_name: Union[Tuple, str] = ..., act_name: Union[Tuple, str] = ..., deep_supervision: bool = False, deep_supr_num: int = 1, res_block: bool = False, trans_bias: bool = False,config=None):
        super().__init__(spatial_dims, in_channels, out_channels, kernel_size, strides, upsample_kernel_size, filters, dropout, norm_name, act_name, deep_supervision, deep_supr_num, res_block, trans_bias)
        batch_size = config['batch_zie']
        self.lin_layer = nn.Sequential(
            nn.Conv3d(32,16,2,2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16,8,2,2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm3d(batch_size),
            linearModel()
        )
    def forward(self, x):
        mid = self.skip_layers(x)
        demo_pred = self.lin_layer(mid) 
        out = self.output_block(mid) 
        if self.training: 
            return out,demo_pred
        else: 
            return out 
