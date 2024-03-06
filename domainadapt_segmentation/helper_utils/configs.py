import argparse
import json
from collections import (
    deque,
)  # just for fun using dequeue instead of just a list for faster appends
from pprint import pprint


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        data_dict = json.load(values)
        arg_list = deque()
        action_dict = {e.option_strings[0]: e for e in parser._actions}
        for i, e in enumerate(data_dict):
            arg_list.extend(self.__build_parse_arge__(e, data_dict, action_dict))
        parser.parse_args(arg_list, namespace=namespace)

    def __build_parse_arge__(self, arg_key, arg_dict, file_action):
        arg_name = f"--{arg_key}"
        arg_val = str(arg_dict[arg_key]).replace(
            "'", '"'
        )  # list of text need to be modified so they can be parsed properly
        try:
            file_action[arg_name].required = False
        except:
            raise KeyError(
                f"The Key {arg_name} is not an expected parameter. Delete it from config or update build_args method in helper_utils.configs.py"
            )
        return arg_name, arg_val


def parse_bool(s: str):
    return eval(s) == True


def warn_optuna(s: str):
    """Take string input of parser for optuna param and just output a warning
    Converts string to boolean for proper reading
    """
    val = eval(s)
    if val:
        print(val)
        print(
            f"WARNING YOU HAVE SELECTED OPTUNA PARAM SEARCH. Most PARAMS WILL BE IGNORED"
        )
    return val


def build_args():
    """Parses args. Must include all hyperparameters you want to tune.

    Special Note:
        Since i entirely expect to load from config files the behavior is
        1.
    """
    parser = argparse.ArgumentParser(
        description="Confguration for my deep learning model training for segmentation"
    )
    parser.add_argument(
        "--data_path",
        required=True,
        type=str,
        help="Path to pickle file containing tuple of form (train_set,val_set,test_set) see Readme for more)",
    )  # TODO: uPDATE README TO EXPLAIN CONFI OF PICKLE FILE
    parser.add_argument(
        "--config_path", required=False, type=open, action=LoadFromFile, help="Path"
    )
    parser.add_argument(
        "--learn_rate",
        required=True,
        type=float,
        help="Initial Learning rate of our model ",
    )
    parser.add_argument(
        "--momentum",
        required=True,
        type=float,
        help="momentum of my optimizer",
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        choices=["2DUnet","3DUnet","2DDinsdaleUnet","2DRamenDinsdale","3DSegRes","3DSegResBias","3DSegResVAE"],
        help="Name of model to be used ",
    )
    parser.add_argument("--epochs", required=True, type=int, help="")
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--spacing_vox_dim", type=json.loads, required=True)
    parser.add_argument("--spacing_pix_dim", type=json.loads, required=True)
    parser.add_argument(
        "--spacing_img_interp", type=str, required=True, choices=["bilinear"]
    )
    parser.add_argument(
        "--spacing_lbl_interp", type=str, required=True, choices=["nearest"]
    )
    parser.add_argument(
        "--scale_intensity_vmin", type=float, required=True, default=-79
    )
    parser.add_argument(
        "--scale_intensity_vmax", type=float, required=True, default=-304
    )
    parser.add_argument("--scale_intensity_bmin", type=float, required=True, default=0)
    parser.add_argument("--scale_intensity_bmax", type=float, required=True, default=1)
    parser.add_argument(
        "--scale_intensity_clip", type=parse_bool, required=True, default=True
    )
    parser.add_argument(
        "--orientation_axcode",
        type=str,
        required=True,
        default="RAS",
        choices=["RAS"],
        help="This is the orientation of the MRI/CT. Careful when selecting",
    )
    parser.add_argument(
        "--device", type=json.loads, required=True, default=["cuda:0"], help="GPU parameter"
    )
    parser.add_argument(
        "--run_param_search",
        type=warn_optuna,
        required=True,
        default=False,
        help="Whehter to run optuna param",
    )
    parser.add_argument(
        "--dev",
        type=parse_bool,
        required=False,
        default=False,
        help="Specify a dev run. Subsamples training data to be just 10% so you can iterate faster",
    )
    parser.add_argument(
        "--num_seg_labels",
        type=int,
        required=True,
        default=2,
        help="Number of segmentation labels. It includes background. i.e if doing foreground vs background  num_seg_labels is 2",
    )
    parser.add_argument(
        "--train_transforms",
        type=json.loads,
        required=True,
        help="List of Names of train transforms and augmentations in form [load,rotate]",
    )
    parser.add_argument(
        "--test_transforms",
        type=json.loads,
        required=True,
        help="List of Names of test transforms and augmentations in form [load] should be subset of train transforms",
    )  # TODO: asert test is subset of train excluding rands
    parser.add_argument("--img_key_name", required=True, type=str)
    parser.add_argument("--lbl_key_name", required=True, type=str)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument('--label_vals',type=json.loads,required=True)
    parser.add_argument(
        "--train_mode",
        type=str,
        required=True,
        choices=["vanilla", "debias", "dinsdale","mixed","consistency",'vae'],
    )
    parser.add_argument("--log_dir", type=str, required=True) 
    parser.add_argument("--2Dvs3D",type=str,required=True,choices=['2D','3D'])
    parser.add_argument('--seed',type=int,default=349)
    parser.add_argument("--resize_size",type=json.loads,required=False)
    parser.add_argument("--model_weight",type=str,required=False)
    add_rand_crop_params(parser)
    add_rand_flip_params(parser)
    add_rand_affine_params(parser)
    add_rand_gauss_params(parser)
    add_rand_shift_params(parser)

    return parser

def build_test_args(): 
    parser = argparse.ArgumentParser(
        description="Confguration for my deep learning model testing for segmentation"
    )
    parser.add_argument('--config-path',required=False,type=open,action=LoadFromFile)
    parser.add_argument('--model_weight',required=True) 
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--device',default='cuda:0',required=False)
    parser.add_argument("--metrics_path",required=True )
    return parser
def build_infer_args(): 
    parser = argparse.ArgumentParser(
        description="Confguration to run inference of my model"
    )
    parser.add_argument('--config-path',required=False,type=open,action=LoadFromFile)
    parser.add_argument('--model_weight',required=False) 
    parser.add_argument('--pkl_path',required=False)
    parser.add_argument('--output_dir',required=False)
    parser.add_argument('--device',default='cuda:0',required=False)
    return parser

def get_params():
    args = build_args()
    my_args = args.parse_args()
    arg_dict = vars(my_args) 
    return arg_dict
def get_test_params(): 
    args = build_test_args() 
    my_args = args.parse_args()
    arg_dict = vars(my_args) 
    return arg_dict 
def get_infer_params():
    args = build_infer_args() 
    my_args = args.parse_args()
    arg_dict = vars(my_args) 
    return arg_dict 


def add_rand_crop_params(parser):
    parser.add_argument(
        "--rand_crop_label_num_samples",
        required=True,
        type=int,
        help="Each Image is cropped into patches. How many random patches should we get for each image. Note batch will be NumberImages*NumberSamples",
    )
    parser.add_argument(
        "--rand_crop_label_positive_samples",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--rand_crop_label_allow_smaller",
        required=True,
        type=parse_bool,
    )


def add_rand_shift_params(parser):
    parser.add_argument(
        "--rand_shift_intensity_offset",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--rand_shift_intensity_prob",
        required=True,
        type=float,
    )


def add_rand_gauss_params(parser):
    parser.add_argument("--rand_gauss_sigma", required=False, type=json.loads)


def add_rand_flip_params(parser: argparse.ArgumentParser):
    parser.add_argument("--rand_flip_prob", required=True, type=parse_bool)


def add_rand_affine_params(parser: argparse.ArgumentParser):
    parser.add_argument("--rand_affine_prob", required=True, type=float)
    parser.add_argument("--rand_affine_rotation_range", required=True, type=json.loads)
    parser.add_argument("--rand_affine_scale_range", required=True, type=json.loads)


if __name__ == "__main__":
    args = build_args()
    my_args = args.parse_args()
    arg_dict = vars(my_args)
    pprint(arg_dict)
