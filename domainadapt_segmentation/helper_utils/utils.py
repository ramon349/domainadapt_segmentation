import random
import math
import os
import numpy as np
from glob import glob
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch


def subsample_list(my_list: list, perc: float):
    num_samples = int(math.floor(len(my_list) * perc))
    return random.sample(my_list, num_samples)


def figure_version(path: str, load_past=False):
    #  when saving model  checkpoints and logs. Need to make sure i don't overwrite previous experiemtns
    avail = glob(f"{path}/run_*")
    if len(avail) == 0:
        ver = "run_0"
    else:
        avail = sorted(avail, key=lambda x: int(x.split("_")[-1]))
        oldest = int(avail[-1].split("_")[-1])
        if load_past:
            ver = f"run_{oldest}"
        else:
            ver = f"run_{oldest+1}"
    return os.path.join(path, ver)


def make_8bit(arr: torch.Tensor):
    arr = arr + arr.min().abs()  #  make 0 the min  value
    return ((torch.maximum(arr, torch.tensor(0)) / arr.max()) * 255).to(torch.uint8)


def proc_batch(imgs: torch.Tensor, labels: torch.Tensor):
    img_l = list()
    lbl_l = list()
    for e in range(imgs.shape[0]):  # iterate trhough batch
        l_idx = labels[e][0].sum(dim=0).sum(dim=0).argmax()
        img_s = (
            imgs[e][0][:, :, l_idx].unsqueeze(0).unsqueeze(0)
        )  # the [0] after e is to account for  channel dimension
        lbl_s = labels[e][0][:, :, l_idx].unsqueeze(0).unsqueeze(0)
        img_l.append(img_s)
        lbl_l.append(lbl_s)
    return torch.cat(img_l, axis=0), torch.cat(lbl_l, axis=0)


def write_batches(writer: SummaryWriter, inputs, labels, epoch,dset=None):
    # do some processing then
    out_imgs, out_lbls = proc_batch(imgs=inputs, labels=labels)
    writer.add_images(f"{dset}_set_img", out_imgs, global_step=epoch)
    writer.add_images(f"{dset}_set_lbl", out_lbls, global_step=epoch)


def show_large_slice(input_dict):
    # TODO: MAKE IT SO I CAN USE THIS IN TENSORBOARD LOGGING
    lbl = input_dict["label"]
    l_idx = (lbl[0] != 0).sum(dim=0).sum(dim=0).argmax()
    lbl_max = lbl[0, :, :, l_idx]
    img = input_dict["image"][0, :, :, l_idx]
    plt.imshow(img, cmap="gray")
    plt.imshow(lbl_max, alpha=0.5)


def dice_score(truth, pred):
    # TODO USE THIS IN TEST PHASE OF FINAL MODEL
    seg = pred.flatten()
    gt = truth.flatten()
    return np.sum(seg[gt == 1]) * 2.0 / (np.sum(seg) + np.sum(gt))


def makeWeightedsampler(ds):
    phase_list = [e["phase"] for e in ds]
    cls_counts = [0, 0]
    cls_counts[0] = len(phase_list) - sum(phase_list)
    cls_counts[1] = sum(phase_list)
    cls_weights = [1 / e for e in cls_counts]
    sample_weight = list()
    for e in phase_list:
        sample_weight.append(cls_weights[e])
    sample_weight = torch.tensor(sample_weight)
    return WeightedRandomSampler(sample_weight, len(sample_weight))

def load_weights(weight_path): 
    ck = torch.load(weight_path,map_location='cpu') 
    conf = ck['conf']
    weights = ck['state_dict'] 
    return conf,weights 