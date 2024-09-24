import random
import math
import os
import numpy as np
from glob import glob
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torch
from collections import OrderedDict
import pdb 
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

def _proc3dbatch(imgs: torch.Tensor, labels: torch.Tensor): 
    img_l = list()
    lbl_l = list()
    for e in range(imgs.shape[0]):  # iterate trhough batch
        l_idx = labels[e][-1].sum(dim=0).sum(dim=0).argmax()
        img_s = (
            imgs[e][-1][:, :, l_idx].unsqueeze(0).unsqueeze(0)
        )  # the [0] after e is to account for  channel dimension
        lbl_s = labels[e][-1][:, :, l_idx].unsqueeze(0).unsqueeze(0)
        img_l.append(img_s)
        lbl_l.append(lbl_s)
    return torch.cat(img_l, axis=0), torch.cat(lbl_l, axis=0)
def _proc2dbatch(imgs:torch.Tensor,labels:torch.Tensor): 
    #TODO: i may need to do some other augmentations in the future. hence 
    return  imgs,  labels


def proc_batch(imgs: torch.Tensor, labels: torch.Tensor,preds = None,config=None):
    if config['2Dvs3D']=='3D': 
        imgs,_labels = _proc3dbatch(imgs,labels) 
        if not (preds is None): 
            preds,_labels = _proc3dbatch(preds,labels) 
        return imgs,_labels,preds
    if config['2Dvs3D']=='2D':
        return imgs,labels,preds 



def write_batches(writer: SummaryWriter, inputs, labels,epoch,preds=None,dset=None,config=None,is_eval=False):
    if is_eval and config['2Dvs3D']=='2D': 
        label_s = labels[0,1,:,:,:] 
        slice_idx = torch.argmax(label_s.sum(axis=0).sum(axis=0))
        inputs = inputs[:,:,:,:,slice_idx] 
        labels = labels[:,:,:,:,slice_idx]  
        if not (preds is None): 
            preds = preds[:,:,:,:,slice_idx]

    if writer: 
        # do some processing then
        out_imgs, out_lbls,preds = proc_batch(imgs=inputs, labels=labels,preds=preds,config=config)
        writer.add_images(f"{dset}_set_img", out_imgs, global_step=epoch)
        writer.add_images(f"{dset}_set_lbl", out_lbls, global_step=epoch)

def write_pred_batches(writer: SummaryWriter, inputs, labels,epoch,preds=None,dset=None,config=None,is_eval=False):
    labels = torch.cat([e.unsqueeze(0) for e in labels]).to('cpu')
    preds = torch.cat([e.unsqueeze(0) for e in preds]).to('cpu')
    if writer: 
        # do some processing then
        out_imgs, out_lbls,preds = proc_batch(imgs=inputs, labels=labels,preds=preds,config=config)
        writer.add_images(f"{dset}_set_img", out_imgs, global_step=epoch)
        writer.add_images(f"{dset}_set_lbl_{1}", out_lbls, global_step=epoch)
        if not (preds is None): 
            writer.add_images(f"{dset}_set_pred_{1}", preds, global_step=epoch)

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
    weights = remove_ddp_tags(ck['model_weight'] )
    return conf,weights 

def remove_ddp_tags(state_d):
    new_d = OrderedDict() 
    for k,v in state_d.items(): 
        new_name = k.replace("module.","") 
        new_d[new_name] = v
    return new_d
import torch.nn as nn
import torch
import numpy as np
#copied form dinsdale repo
class confusion_loss(nn.Module):
    def __init__(self, task=0,reduction=None):
        super(confusion_loss, self).__init__()
        self.task = task
        self.reduction = reduction

    def forward(self, x, target):
        eps = 1e-10
        # We only care about x
        log = torch.log(nn.functional.softmax( x) + eps)
        log_sum = torch.sum(log, dim=1)
        normalised_log_sum = torch.div(log_sum,  x.size()[1])
        if self.reduction=='sum': 
            loss = torch.mul(torch.sum(normalised_log_sum, dim=-1),-1) 
        if self.reduction =='mean':
            loss = torch.mul(torch.mean(normalised_log_sum),-1) 
        if self.reduction =='none':
            loss = torch.mul(normalised_log_sum,-1)
        return loss 