from monai.networks.nets.unet import UNet
from monai.networks.nets.dynunet import DynUNet 
from .dinsdale import RamenDinsdale2D


def model_factory(config):
    model_name = config["model"]
    num_seg_labels = config["num_seg_labels"]
    if model_name == "2DUnet":
        net = UNet(
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
        net = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_seg_labels,
            channels=(16, 32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2, 2),
            num_res_units=2,
            act="LEAKYRELU",
        )
        return net
    if model_name=='2DDinsdale':
        net = RamenDinsdale2D(1,2)
        return net
