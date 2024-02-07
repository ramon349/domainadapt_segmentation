from monai.transforms import (
    Compose,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    EnsureChannelFirstd,
    RandCropByPosNegLabeld,
    Spacingd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianSmoothd,
    Orientationd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandAffined,
    FillHolesd,
    SpatialPadd,
    LabelToMaskd,
)
from monai.data import NibabelReader
import torch
from monai.inferers import sliding_window_inference
import matplotlib.pyplot as plt
from glob import glob
import os
from monai.transforms.croppad.batch import PadListDataCollate, replace_element
import numpy as np
from monai.transforms.croppad.array import CenterSpatialCrop
from monai.data.utils import list_data_collate


def get_transform(name, conf,mode='train'):
    img_k = conf["img_key_name"]
    lbl_k = conf["lbl_key_name"]
    if mode=='train': 
        transform_ins = [img_k,lbl_k]
    else: 
        transform_ins = [img_k]
    if img_k == lbl_k:
        raise AssertionError("Image Key and Label Key should not be the same")
    if name == "load":
        # for now reader will only be the nibabel one
        # TODO make it interchangeable with the GCP one
        return LoadImaged(keys=transform_ins,reader=NibabelReader)
    if name == "channel_first":
        return EnsureChannelFirstd(keys=transform_ins)
    if name == "scale_intensity":
        vmin = conf["scale_intensity_vmin"]
        vmax = conf["scale_intensity_vmax"]
        bmin = conf["scale_intensity_bmin"]
        bmax = conf["scale_intensity_bmax"]
        clip = conf["scale_intensity_clip"]
        return ScaleIntensityRanged(
            keys=img_k, a_min=vmin, a_max=vmax, b_min=bmin, b_max=bmax, clip=clip
        )
    if name == "crop_foreground":
        return CropForegroundd(keys=transform_ins, source_key=img_k)
    if name == "orient":
        ax_code = conf["orientation_axcode"]
        return Orientationd(keys=transform_ins, axcodes=ax_code)
    if name == "spacing":
        pix_dim = conf["spacing_pix_dim"]
        vox_dim = conf["spacing_vox_dim"]
        img_interp = conf["spacing_img_interp"]
        lbl_interp = conf["spacing_lbl_interp"]
        if len(transform_ins)==2: 
            interp_mode= (img_interp,lbl_interp)
        else: 
            interp_mode = (img_interp,)
        return Spacingd(
            keys=transform_ins, pixdim=pix_dim, mode=interp_mode
        )
    if name == "rand_crop_label":
        vox_dim = conf["spacing_vox_dim"]
        num_samples = conf["rand_crop_label_num_samples"]
        ps_samples = conf["rand_crop_label_positive_samples"]
        allow_smaller = conf["rand_crop_label_allow_smaller"]
        neg_samples = 1 - ps_samples
        assert ps_samples < 1
        return RandCropByPosNegLabeld(
            keys=transform_ins,
            spatial_size=vox_dim,
            num_samples=num_samples,
            label_key=lbl_k,
            pos=ps_samples,
            neg=neg_samples,
            allow_smaller=allow_smaller,
        )
    if name == "spatial_pad":
        vox_dim = conf["spacing_vox_dim"]
        return SpatialPadd(keys=transform_ins, spatial_size=vox_dim)
    if name == "rand_shift_intensity":
        offset = conf["rand_shift_intensity_offset"]
        prob = conf["rand_shift_intensity_prob"]
        return RandShiftIntensityd(keys=[img_k], offsets=offset, prob=prob)
    if name == "rand_gauss":
        # TODO: SEARCH FOR REASONING TO HAVE VARIABLE X,Y,Z. I guess it would pick up on extra noise from resampling volumes?
        sigma_x = conf["rand_gauss_sigma"]
        return RandGaussianSmoothd(
            keys=[img_k], sigma_x=sigma_x, sigma_y=sigma_x, sigma_z=sigma_x
        )
    if name == "rand_flip":
        prob = conf["rand_flip_prob"]
        return RandFlipd(keys=transform_ins, prob=prob)
    if name == "rand_affine":
        img_interp = conf["spacing_img_interp"]
        lbl_interp = conf["spacing_lbl_interp"]
        affine_prob = conf["rand_affine_prob"]
        rotation_range = conf["rand_affine_rotation_range"]
        scale_range = conf["rand_affine_scale_range"]
        return RandAffined(
            keys=[img_k, lbl_k],
            mode=[img_interp, lbl_interp],
            prob=affine_prob,
            scale_range=scale_range,
            rotate_range=rotation_range,
        )
    if name =='labelMask':
        label_vals = conf['label_vals']
        print(label_vals)
        return LabelToMaskd(select_labels=label_vals,keys=[lbl_k],merge_channels=False)
    raise ValueError(
        f"The param name {name} does not have  a match check typo in config or update transforms.get_transform.py"
    )


def gen_transforms(confi):
    train_transform = Compose(
        [get_transform(e, confi) for e in confi["train_transforms"]]
    )
    val_transform = Compose([get_transform(e, confi) for e in confi["test_transforms"]])
    return train_transform, val_transform


def gen_test_transforms(confi):
    my_transforms = list() 
    for e in confi['test_transforms']:
        l_transform = get_transform(e,confi,mode='test')
        print(l_transform)
        my_transforms.append(l_transform)
    val_transform = Compose(my_transforms) #Compose([get_transform(e, confi) for e in confi["test_transforms"]])
    return val_transform


class ramonPad(PadListDataCollate):
    """Padidng method that simply  adds extra zeros
    TODO:  Do not understand why default monai doesn't have this. Cary over of past
    """

    def __init__(self):
        super().__init__()

    def __call__(self, batch):
        """
        Args:
            batch: batch of data to pad-collate
        """
        # data is either list of dicts or list of lists
        is_list_of_dicts = isinstance(batch[0], dict)
        # loop over items inside of each element in a batch
        batch_item = (
            tuple(batch[0].keys()) if is_list_of_dicts else range(len(batch[0]))
        )
        for key_or_idx in batch_item:
            # calculate max size of each dimension
            max_shapes = []
            for elem in batch:
                if not isinstance(elem[key_or_idx], (torch.Tensor, np.ndarray)):
                    break
                max_shapes.append(elem[key_or_idx].shape[1:])
            # len > 0 if objects were arrays, else skip as no padding to be done
            if not max_shapes:
                continue
            # If all same size, skip
            max_shape = np.array(max_shapes).max(axis=0)

            if np.all(np.array(max_shapes).min(axis=0) == max_shape):
                continue
            for i in range(len(max_shape)):
                if max_shape[i] % 2 != 0:
                    max_shape[i] = max_shape[i] + 1

            # Use `SpatialPad` to match sizes, Default params are central padding, padding with 0's
            padder = SpatialPad(
                spatial_size=max_shape,
                method=self.method,
                mode=self.mode,
                **self.kwargs,
            )
            for idx, batch_i in enumerate(batch):
                orig_size = batch_i[key_or_idx].shape[1:]
                padded = padder(batch_i[key_or_idx])
                batch = replace_element(padded, batch, idx, key_or_idx)

                # If we have a dictionary of data, append to list
                # padder transform info is re-added with self.push_transform to ensure one info dict per transform.
                if is_list_of_dicts:
                    self.push_transform(
                        batch[idx],
                        key_or_idx,
                        orig_size=orig_size,
                        extra_info=self.pop_transform(
                            batch[idx], key_or_idx, check=False
                        ),
                    )

        # After padding, use default list collator
        return list_data_collate(batch)
