import torch
from .helper_utils import configs as help_configs 
from .helper_utils import  utils as help_utils 
from .helper_utils import transforms as help_transforms
from .models.model_factory import model_factory
from torch.utils.data import DataLoader
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
    Activations
)
from monai.transforms import Invertd, SaveImaged, RemoveSmallObjectsd
from hashlib import sha224
import pandas as pd 
from monai.metrics import DiceMetric 
import numpy as np 
import os 
import pdb 

torch.multiprocessing.set_sharing_strategy('file_system')

def subject_formater(metadict,self):
    pid = metadict['filename_or_obj']
    out_form=sha224(pid.encode('utf-8')).hexdigest()
    return {'subject':f"{out_form}","idx":"0"}
def make_post_transforms(test_conf,test_transforms):
    out_dir = test_conf["output_dir"]
    bin_preds = True #TODO: is it woth having continious outputs 
    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                orig_meta_keys ='image_meta_dict',
                meta_keys="pred_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            AsDiscreted(keys="pred",argmax=True),
            RemoveSmallObjectsd(keys="pred", min_size=500),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=out_dir,
                output_postfix="seg",
                resample=False,
                data_root_dir="",
                output_name_formatter=subject_formater,
                savepath_in_metadict=True
            ),
        ]
    )
    return post_transforms

def eval_loop(model, loader,  config,device='cuda:0',post_t=None):
    roi_size = config["spacing_vox_dim"]
    img_k = config["img_key_name"]
    lbl_k = config["lbl_key_name"]
    device = device
    num_seg_labels = config["num_seg_labels"]
    metric = DiceMetric(include_background=True, reduction="mean")
    model.eval()
    all_losses = list()
    dice_scores = list()
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([Activations(to_onehot=2)])
    _step = 0 
    pids = list() 
    img_path = list() 
    lbl_path = list() 
    with torch.no_grad():
        for val_data in tqdm(loader, total=len(loader)):
            val_inputs, val_labels = (
                val_data[img_k].to(device),
                val_data[lbl_k].to(device),
            )
            val_data['pred'] = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
            )
            val_outputs = [post_pred(i).to('cpu') for i in decollate_batch(val_data['pred'])]
            val_labels = [post_label(i).to('cpu') for i in decollate_batch(val_labels)]
            val_store = [post_t(i) for i in decollate_batch(val_data)]

            metric(y_pred=val_outputs, y=val_labels)
            metric_val = metric.aggregate().item()
            metric.reset()
            dice_scores.append(metric_val)
            pids.append(val_data['pid'][0])
            img_path.append(val_data['image_meta_dict']['filename_or_obj'])
            lbl_path.append(val_data['label_meta_dict']['filename_or_obj'])
        out_df = pd.DataFrame({'pids':pids,'img':img_path,'lbl':lbl_path,'dice':dice_scores}) 
    return out_df 

def test_main():
    config = help_configs.get_test_params() 
    weight_path = config['model_weight'] 
    output_dir = config['output_dir']
    device= config['device']
    train_conf, weights = help_utils.load_weights(weight_path=weight_path)
    model= model_factory(config=train_conf) 
    model.load_state_dict(weights)
    if config['test_set']: 
        print(f"Using provided test set: {config['test_set']} ")
        with open(config['test_set'],'rb') as f: 
            test = pkl.load(f)
            test = test[-1] 
    else: 
        with open(train_conf['data_path'],'rb' ) as f : 
            test = pkl.load(f) 
            test = test[-1] # TODO: DON'T KEEP THIS FOREVER 
    dset = kit_factory('basic') # dset that is not cached 
    test_t = help_transforms.gen_test_transforms(confi=train_conf)
    test_ds = dset(test,transform=test_t)
    post_t= make_post_transforms(config,test_t)
    test_loader = DataLoader(test_ds,
        batch_size = 1,
        shuffle=False,
        num_workers = 16,
        collate_fn = help_transforms.ramonPad())
    model = model.to(device=device)
    model.eval() 
    roi_size = train_conf['spacing_vox_dim']
    sw_batch_size=  train_conf['rand_crop_label_num_samples']
    f_name = train_conf['log_dir'].split('/')[-1]
    results_path =  config['metrics_path']
    with torch.no_grad(): 
       outs = eval_loop(model,test_loader,config=train_conf,device=device,post_t=post_t)
       outs.to_csv(results_path,index=False)
if __name__ =='__main__': 
    test_main() 
