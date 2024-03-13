from monai.networks.nets.dynunet import DynUNet 
from monai.networks.nets.unet import Unet as monaiUNet
from monai.networks.nets.segresnet import SegResNet,SegResNetVAE 
from .dinsdale import RamenDinsdale2D
from .dinsdale import UNet as Dinsdale2DUnet
from .ramenUnet import segResnetBias,SegResnetBiasClassiOneBranch,SegResnetBiasClassiTwoBranch,SegResVAE
import pdb 
import torch 
from collections import OrderedDict 
from ..helper_utils.utils import remove_ddp_tags
import pdb 
def get_kernels_strides(patch_size,spacing):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = patch_size, spacing
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def model_factory(config):
    model_name = config["model"]
    num_seg_labels = config["num_seg_labels"]
    if model_name=='unet':
        net =  monaiUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_seg_labels,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            act="LEAKYRELU",
        )
    if model_name == "2DUnet":
        net = DynUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=num_seg_labels,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            act="LEAKYRELU",
        )
    if model_name =='3DUnet':
        kernels,strides = get_kernels_strides(config['spacing_vox_dim'],config['spacing_pix_dim'])
        net = monaiUNet(spatial_dims=3,in_channels=1,out_channels=config['num_seg_labels'],  channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            act="LEAKYRELU",
        )
    if model_name=='2DDinsdaleUnet':
        net = Dinsdale2DUnet(1,2)
    if model_name=='2DRamenDinsdale':
        net = RamenDinsdale2D(1,2)
    if model_name=='3DSegRes':
        net = SegResNet(spatial_dims=3,in_channels=1,out_channels=num_seg_labels)
    if model_name =='3DSegResBias':
        net = segResnetBias(spatial_dims=3,out_channels=num_seg_labels)
    if model_name=='3DSegResBiasClassOne':
        net = SegResnetBiasClassiOneBranch(spatial_dims=3,out_channels=num_seg_labels)
    if model_name=='3DSegResBiasClassTwo':
        net =  SegResnetBiasClassiTwoBranch(spatial_dims=3,out_channels=num_seg_labels)
    if model_name=='3DSegResVAE':
        net = SegResVAE(input_image_size=(128,128,128),spatial_dims=3,vae_estimate_std=True)
    if 'model_weight' in config and config['model_weight']: 
        print('loading weights')
        checkpoint= torch.load(config['model_weight'],map_location='cpu') 
        new_d = remove_ddp_tags(checkpoint['state_dict']) 
        net.load_state_dict(new_d,strict=False)
    return net 
