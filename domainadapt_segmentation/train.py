import torch
import os
import random
import os
from .helper_utils import configs as help_configs
from .helper_utils import data_io as help_io
from .helper_utils import transforms as help_transforms
from .helper_utils import utils as help_utils
from .batch_iterators.train_iterators import *
from .data_factories.kits_factory import kit_factory
from monai.data import SmartCacheDataset, partition_dataset
from .models.model_factory import model_factory
from monai.losses import DiceCELoss
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import optuna
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from domainadapt_segmentation.helper_utils.utils import confusion_loss
from torch.nn import CosineEmbeddingLoss
import numpy as np
import pdb
import torch._dynamo
from torch.utils.data import WeightedRandomSampler
from .batch_iterators.trainer_factory import load_trainer

torch.multiprocessing.set_sharing_strategy("file_system")


def makeWeightedsampler(ds):
    classes = [0, 1]
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


def optuna_gen(conf_in, trial):
    pass


def _parse():
    conf = help_configs.get_params()
    gpus = conf["device"]
    world_size = len(gpus)
    if world_size == 1:
        dummy_main(0, 1, conf)
    else:
        mp.spawn(dummy_main, args=(world_size, conf), nprocs=world_size, join=True)


def get_data_laoders(conf):
    train, val, test = help_io.load_data(
        conf["data_path"]
    )  # this is just a list of dictionaries
    train_transform, val_transform = help_transforms.gen_transforms(
        conf
    )  # no changes needed for default transforms
    dset = kit_factory("cached")  # i do caching when i shouldn't
    batch_size = conf["batch_size"]
    cache_dir = conf["cache_dir"]
    if conf["rank"] >= 0:
        rank = conf["rank"]
        tr_part = partition_dataset(
            train,
            num_partitions=dist.get_world_size(),
            shuffle=False,
            even_divisible=True,
            drop_last=False,
        )[
            rank
        ]  # this will create a dataset  for each partion
        val_part = partition_dataset(
            val,
            num_partitions=dist.get_world_size(),
            shuffle=False,
            even_divisible=True,
            drop_last=False,
        )[rank]
    else:
        tr_part = train
        val_part = val
    print(f"With rank {rank} has {len(tr_part)}")
    tr_dset = dset(tr_part, transform=train_transform, cache_dir=cache_dir)
    val_dset = dset(val_part, transform=val_transform, cache_dir=cache_dir)
    test_dset = dset(test, transform=val_transform, cache_dir=cache_dir)
    if conf["balance_phases"]:
        print("PHASE BALANCE IS ENABLED")
        sampler = makeWeightedsampler(tr_part)
        shuffle = False
    else:
        print("PHASE BALANCE IS NOT ENABLED")
        sampler = None
        shuffle = True
    num_workers = conf["num_workers"]
    data_loaders = dict()
    data_loaders["train"] = DataLoader(
        tr_dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=help_transforms.ramonPad(),
    )
    data_loaders["val"] = DataLoader(
        val_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=help_transforms.ramonPad(),
    )
    data_loaders["test"] = DataLoader(
        test_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=help_transforms.ramonPad(),
    )
    return data_loaders


def make_writer(conf):
    log_path = help_utils.figure_version(
        conf["log_dir"], load_past=conf["resume"]
    )  # TODO think about how you could perhaps continue training
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_dir=log_path)
    return writer, log_path


def dummy_main(rank, world_size, conf):
    """This is the actual main function used during training
    rank: is the rank order in distributed training. On local training it defaults to 0
    world_size: is the number of gpus being used. On local mode it is defaulted to 1
    """
    seed = conf["seed"]
    setup_repro(seed)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method="env://"
    )  # this init method was suggested by tutorials on monai
    rank = dist.get_rank()
    cuda_str = conf["device"][rank]
    device = torch.device(cuda_str)
    conf["rank"] = rank
    # torch.cuda.set_device(device)
    if rank == 0:
        writer, log_path = make_writer(conf)
        conf["log_dir"] = log_path
    else:
        writer, log_path = None, None
    model = model_factory(config=conf)
    model = model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[device])
    data_loaders = get_data_laoders(conf)
    trainer_func = load_trainer(conf["train_mode"])
    trainer = trainer_func(model, tb_writter=writer, conf=conf, dl_dict=data_loaders)
    trainer._log_model_graph()
    trainer.fit()


def setup_repro(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    _parse()
