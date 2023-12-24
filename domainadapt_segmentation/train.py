import torch
from monai.data import DataLoader
import os
import random
import os
import helper_utils.configs as help_configs
import helper_utils.data_io as help_io
import helper_utils.transforms as help_transforms
import helper_utils.utils as help_utils
from batch_iterators.train_iterators import *
from data_factories.kits_factory import kit_factory
from monai.data import DataLoader
from models.model_factory import model_factory
from monai.losses import DiceCELoss
import torch._dynamo
import optuna

torch._dynamo.config.suppress_errors = True
torch.multiprocessing.set_sharing_strategy("file_system")


def optuna_gen(conf_in, trial):
    pass


def _parse():
    conf = help_configs.get_params()
    if conf["run_param_search"]:
        # setup optuna exp
        model_name = f"{conf['train_mode']}_{conf['model']}"
        storage_name = f"sqlite:///media/Datacenter_storage/ramon_dataset_curations/domainadapt_segmentation/optuna_logs/{model_name}.db"
        study_name = "loss_search"
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            load_if_exists=True,
        )
        objective = lambda x: main(conf, x)
        unique_trials = 100
        if unique_trials > len(study.trials):
            study.optimize(objective, n_tirals=100)
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))
    else:
        main(conf)


def main(conf_in, trial=None):
    if trial:
        conf = optuna_gen(copy(conf_in), trial)
    else:
        conf = conf_in
    dset = kit_factory("cached")
    train, val, test = help_io.load_data(conf["data_path"])
    # use short circuitting to check if dev is  a field
    if "dev" in conf.keys() and conf["dev"] == True:
        print(
            "we are outputting to devset we are therefore using a smaller train sample for dev"
        )
        train = random.sample(train, 50)
        val = random.sample(val, 30)
    print(f"Len train is {len(train)}")
    print(f"Len val is {len(val)}")
    print(f"Len test is {len(test)}")
    print(conf)
    lr = conf["learn_rate"]
    momentum = conf["momentum"]
    train_transform, val_transform = help_transforms.gen_transforms(conf)
    batch_size = conf["batch_size"]
    cache_dir = conf["cache_dir"]
    os.makedirs(cache_dir, exist_ok=True)
    train_ds = dset(train, transform=train_transform, cache_dir=cache_dir)
    val_ds = dset(val, transform=val_transform, cache_dir=cache_dir)
    test_ds = dset(test, transform=val_transform, cache_dir=cache_dir)
    num_workers = conf["num_workers"]
    if conf["train_mode"] == "debias" or conf["train_mode"] == "mixed":
        sampler = help_utils.makeWeightedsampler(train)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=help_transforms.ramonPad(),
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=help_transforms.ramonPad(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=help_transforms.ramonPad(),
    )
    loaders = (train_loader, val_loader, test_loader)
    DEVICE = torch.device(conf["device"])
    model = model_factory(config=conf)
    if "pretrained" in conf.keys():
        # TODO add support for continuing training by providing optinal path to checkpoint
        ck = torch.load(conf["pretrained"])
        model.load_state_dict(ck["state_dict"])
        print("state dict loaded")
    model = model.to(torch.float32).to(DEVICE)
    # model = torch.compile(model, fullgraph=False, dynamic=True)

    # TODO: make the dice metric and loss function modifiable
    loss_function = DiceCELoss(
        include_background=True, reduction="mean", to_onehot_y=True, softmax=True
    )
    if conf["train_mode"] == "vanilla" or conf['train_mode']=='mixed':
        # lr should be 0.01 for these experiments
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum)
        lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer,
            total_iters=conf["epochs"],
        )
        val_loss = train_batch(
            model,
            loaders,
            optimizer,
            lr_scheduler,
            loss_function,
            device=DEVICE,
            config=conf,
        )
    #TODO: ADD TESTING OF THE BEST MODEL 


if __name__ == "__main__":
    _parse()
