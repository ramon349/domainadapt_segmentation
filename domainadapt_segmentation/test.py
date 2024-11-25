import torch
from .helper_utils import configs as help_configs
from .helper_utils import utils as help_utils
from .helper_utils import transforms as help_transforms
from .models.model_factory import model_factory
from torch.utils.data import DataLoader
from .data_factories.kits_factory import kit_factory
import pickle as pkl

## unknown imports
import torch
import pickle as pkl
from .data_factories.kits_factory import kit_factory
from monai.inferers import sliding_window_inference
from monai.data import (
    decollate_batch,
)  # this is needed wherever i run the iterator
from tqdm import tqdm
from monai.transforms import Invertd, SaveImaged, RemoveSmallObjectsd
from hashlib import sha224
import pandas as pd
from monai.metrics import DiceMetric
import numpy as np
import os
import pdb

torch.multiprocessing.set_sharing_strategy("file_system")


from .batch_iterators.trainer_factory import load_trainer
from .helper_utils.transforms import make_post_transforms


def test_main():
    config = help_configs.get_test_params()
    config["rank"] = 0
    # load the model
    weight_path = config["model_weight"]

    train_conf, weights = help_utils.load_weights(weight_path=weight_path)
    train_conf["device"] = config["device"]
    train_conf["rank"] = config["rank"]
    model = model_factory(config=train_conf)
    model.load_state_dict(weights)
    device = config["device"][0]
    model = model.to(device)
    # load the test_pkl of interest
    with open(config["test_set"], "rb") as f:
        test = pkl.load(f)
        test = test[-1]
    dset = kit_factory("basic")  # dset that is not cached
    # get the relevant transforms
    test_t = help_transforms.gen_test_transforms(confi=train_conf)
    test_ds = dset(test, transform=test_t)
    post_t = make_post_transforms(config, test_t)
    dl = dict()
    dl["test"] = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=16,
        collate_fn=help_transforms.ramonPad(),
    )
    trainer = load_trainer(config["trainer"])(
        model, tb_writter=None, dl_dict=dl, conf=train_conf
    )
    outs = trainer.test_loop(dl["test"], post_t)
    results_path = config["metrics_path"]
    outs.to_csv(results_path, index=False)


if __name__ == "__main__":
    test_main()
