from .helper_utils import configs as help_configs
from .helper_utils import utils as help_utils
from .helper_utils import transforms as help_transforms
from .models.model_factory import model_factory
from monai.data import DataLoader
from .data_factories.kits_factory import kit_factory
import pickle as pkl
import torch
import pickle as pkl
from .data_factories.kits_factory import kit_factory
from .test import make_post_transforms
from .batch_iterators.trainer_factory import load_trainer


def infer_main():
    config = help_configs.get_infer_params()
    config["rank"] = 0
    weight_path = config["model_weight"]
    train_conf, weights = help_utils.load_weights(weight_path=weight_path)
    train_conf["device"] = config["device"]
    train_conf["rank"] = config["rank"]
    device = config["device"][0]
    model = model_factory(train_conf)
    model.load_state_dict(weights)
    model = model.to(device)
    with open(config["pkl_path"], "rb") as f:
        test = pkl.load(f)
        if len(test)==3: 
            test = test[-1]
        else:
            test = test  # TODO: DON'T KEEP THIS FOREVER
    dset = kit_factory("basic")  # dset that is not cached
    test_t = help_transforms.gen_test_transforms(confi=train_conf, mode="infer")
    test_ds = dset(test, transform=test_t)
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=help_transforms.ramonPad(),
    )
    dl_dict = {"infer": test_loader}
    model = model.to(device=device)
    model.eval()
    post_transform = make_post_transforms(config, test_transforms=test_t)
    trainer = load_trainer(config["trainer"])(
        model, tb_writter=None, dl_dict=dl_dict, conf=train_conf
    )
    outs = trainer.infer_loop(post_transform)
    results_path = config["mapping_path"]
    outs.to_csv(results_path, index=False)


if __name__ == "__main__":
    infer_main()
