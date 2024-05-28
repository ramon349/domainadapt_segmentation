
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
from .test import subject_formater
import pdb 
from monai.data import Dataset 
from monai.data.image_writer import ITKWriter
import os 

def make_post_transforms(test_conf,test_transforms):
    out_dir = test_conf["output_dir"]
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
                savepath_in_metadict=True,
            ),
        ]
    )
    return post_transforms

def make_mapping(input_d,conf):
    og_paths = list() 
    seg_paths = list() 
    for e in input_d['filename_or_obj']:
        new_name =  subject_formater(e) 
        original_path =e 
        out_dir = conf['output_dir'] 
        full_path =  os.path.join(out_dir,new_name,new_name,'_seg_0.nii.gz')
        og_paths.append(original_path) 
        seg_paths.append(full_path)

    return  (og_paths,seg_paths)

def infer_loop(model, loader,  config,post_transform=None,output_conf=None):
    roi_size = config["spacing_vox_dim"]
    img_k = config["img_key_name"]
    device = config["device"]
    num_seg_labels = config["num_seg_labels"]
    model.eval()
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_seg_labels)])
    post_label = Compose([AsDiscrete(to_onehot=num_seg_labels)])
    pids = list() 
    img_paths= list() 
    seg_paths = list() 
    dicts = list() 
    with torch.no_grad():
        for val_data in tqdm(loader, total=len(loader)):
            val_inputs= (
                val_data[img_k]
            )
            val_data['pred'] = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=8,
                predictor=model,
                sw_device=device,
                device='cpu',
            )
            post_transform(decollate_batch(val_data)[0])
            orig_path ,seg_path =  make_mapping(val_data['image_meta_dict'],output_conf)
            img_paths.extend(orig_path)
            seg_paths.extend(seg_path)
            out_df = pd.DataFrame({'input_img':img_paths,'output_seg':seg_paths})
            out_df.to_csv("./temp_seg_log.csv",index=False)
    return out_df

def main():
    config = help_configs.get_infer_dcm_params() 
    weight_path = config['model_weight'] 
    device= config['device']
    train_conf, weights = help_utils.load_weights(weight_path=weight_path)
    model= model_factory(config=train_conf) 
    model.load_state_dict(weights)
    with open(config['infer_set'],'rb' ) as f : 
        test = pkl.load(f) 
    test_t = help_transforms.gen_test_transforms(confi=train_conf,mode='infer')
    test_ds = Dataset(test,transform=test_t,)
    test_loader = DataLoader(test_ds,
    batch_size = 1,
    shuffle=False,
    num_workers = 8,
    collate_fn = help_transforms.ramonPad(),)
    model = model.to(device=device)
    model.eval() 
    post_transform = make_post_transforms(config,test_transforms=test_t)
    train_conf['device'] = device
    sw_batch_size=  1 
    with torch.no_grad(): 
       outs = infer_loop(model,test_loader,config=train_conf,post_transform=post_transform,output_conf=config)
       f_name = config['seg_map_path']
       outs.to_csv(f_name,index=False)


if __name__=='__main__':
    main()
