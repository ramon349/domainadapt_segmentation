from monai.networks.nets.dynunet import DynUNet 
from monai.networks.nets.unet import Unet as monaiUNet
from .dinsdale import RamenDinsdale2D
from .dinsdale import UNet as Dinsdale2DUnet
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
        return net
    if model_name =='3DUnet':
        kernels,strides = get_kernels_strides(config['spacing_vox_dim'],config['spacing_pix_dim'])
        net = monaiUNet(spatial_dims=3,in_channels=1,out_channels=1,  channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            act="LEAKYRELU",
        )

        """
         net = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_seg_labels,
            kernel_size=kernels,
            strides=strides,
            res_block=True,
            upsample_kernel_size=strides[1:],
            norm_name="INSTANCE")
        
        """
        return net
    if model_name=='2DDinsdaleUnet':
        net = Dinsdale2DUnet(1,2)
        return net
    if model_name=='2DRamenDinsdale':
        net = RamenDinsdale2D(1,2)
        return net
