
import torch
from .helper_utils import configs as help_configs 
from .helper_utils import  utils as help_utils 
from .helper_utils import transforms as help_transforms
from .models.model_factory import model_factory
from monai.data import DataLoader
from .data_factories.kits_factory import kit_factory
import pickle  as pkl 
## unknown imports 
import torch
import pickle as pkl
from .data_factories.kits_factory import kit_factory
from monai.inferers import sliding_window_inference
from monai.data import (
    decollate_batch,
)  # this is needed wherever i run the iterator
from tqdm import tqdm
from monai.transforms import (
    Compose,
    AsDiscrete,
    AsDiscreted,
    Activationsd
)
from monai.transforms import Invertd, SaveImaged, RemoveSmallObjectsd
from hashlib import sha224
import pandas as pd 
from monai.metrics import DiceMetric 
import numpy as np 
from .test import make_post_transforms,eval_loop
import pdb 
def infer_loop(model, loader,  config,post_transform=None):
    roi_size = config["spacing_vox_dim"]
    img_k = config["img_key_name"]
    device = config["device"]
    num_seg_labels = config["num_seg_labels"]
    model.eval()
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_seg_labels)])
    post_label = Compose([AsDiscrete(to_onehot=num_seg_labels)])
    pids = list() 
    img_path = list() 
    seg_path = list() 
    dicts = list() 
    with torch.no_grad():
        for val_data in tqdm(loader, total=len(loader)):
            val_inputs= (
                val_data[img_k].to(device)
            )
            val_data['pred'] = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
            )
            out_meta = post_transform(decollate_batch(val_data)[0])
            img_path.append(val_data['image_meta_dict']['filename_or_obj'])
    return img_path

def infer_main():
    config = help_configs.get_infer_params() 
    weight_path = config['model_weight'] 
    output_dir = config['output_dir']
    device= config['device']
    train_conf, weights = help_utils.load_weights(weight_path=weight_path)
    model= model_factory(config=train_conf) 
    model.load_state_dict(weights)

    with open(config['pkl_path'],'rb' ) as f : 
        test = pkl.load(f) 
        test = test[-1] # TODO: DON'T KEEP THIS FOREVER 
    dset = kit_factory('basic') # dset that is not cached 
    test_t = help_transforms.gen_test_transforms(confi=train_conf,mode='infer')
    test_ds = dset(test,transform=test_t)
    test_loader = DataLoader(test_ds,
    batch_size = 1,
    shuffle=False,
    num_workers = 8,
    collate_fn = help_transforms.ramonPad(),)
    model = model.to(device=device)
    model.eval() 
    post_transform = make_post_transforms(config,test_transforms=test_t)
    roi_size = train_conf['spacing_vox_dim']
    train_conf['device'] = device
    sw_batch_size=  1 
    with torch.no_grad(): 
       outs = infer_loop(model,test_loader,config=train_conf,post_transform=post_transform)
       f_name = train_conf['log_dir'].split('/')[-1]
       outs.to_csv(f'../results/{f_name}_test_set.csv',index=False)

if __name__=='__main__':
    infer_main()