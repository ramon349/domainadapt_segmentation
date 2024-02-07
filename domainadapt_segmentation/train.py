import torch
from monai.data import DataLoader
import os
import random
import os
from   .helper_utils import configs as help_configs
from  .helper_utils import data_io as help_io
from .helper_utils import transforms as help_transforms
from .helper_utils import utils as help_utils
from .batch_iterators.train_iterators import *
from .data_factories.kits_factory import kit_factory
from monai.data import DataLoader,SmartCacheDataset,partition_dataset
from .models.model_factory import model_factory
from monai.losses import DiceCELoss
import torch._dynamo
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import optuna
import os 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel 
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
        gpus = conf['device']  
        print(gpus)
        for e in gpus: 
            print(e.split(':'))
        device_nums = [e.split(':')[1] for e in gpus]
        os.environ['CUDA_VISIBLE_DEVICES']= ",".join(device_nums) 
        world_size = len(gpus)
        mp.spawn(dummy_main,args=(world_size,conf),nprocs=world_size,join=True)
        #main(conf)

def reduce_tensors(tensor,op=dist.ReduceOp.SUM,world_size=2): 
    tensor = tensor.clone() 
    dist.all_reduce(tensor,op) 
    tensor.div(world_size) 
    return tensor
def dummy_main(rank,world_size,conf):
    print(f"Hello I am procees {rank} out of {world_size}")
    print(f"Process {rank} has {torch.cuda.device_count()} gpus avaialble") 
    seed = conf['seed']
    setup_repro(seed)
    dist.init_process_group(backend='nccl',rank=rank,world_size=world_size,init_method="env://") 
    #here we set up the partitioning of the training data across all the workers 
    ##SETUP ALL THE DATASET RELATED ITEMS 
    train, val, test = help_io.load_data(conf["data_path"]) # this is just a list of dictionaries  
    # use short circuitting to check if dev is  a field
    if "dev" in conf.keys() and conf["dev"] == True:
        print(
            "we are outputting to devset we are therefore using a smaller train sample for dev"
        )
        train = random.sample(train, 50)
        val = random.sample(val, 30)
    train_transform, val_transform = help_transforms.gen_transforms(conf) # no changes needed for default transforms 
    tr_part = partition_dataset(train,num_partitions=dist.get_world_size(),shuffle=True,even_divisible=True) [dist.get_rank()] # this will create a dataset  for each partion 
    val_part = partition_dataset(val,num_partitions=dist.get_world_size(),shuffle=True,even_divisible=True)[dist.get_rank()] 
    dset = kit_factory('cached')
    batch_size = conf['batch_size']
    cache_dir = conf['cache_dir']
    tr_dset = dset(tr_part,transform= train_transform,cache_dir=cache_dir)
    val_dset = dset(val_part,transform=val_transform,cache_dir=cache_dir)
    num_workers = conf['num_workers']
    train_dl = DataLoader(
        tr_dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=help_transforms.ramonPad(),
        sampler=None,
    )
    val_dl = DataLoader(
        val_dset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=help_transforms.ramonPad(),
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device) 
    #load the model 
    model = model_factory(config=conf)
    loss_function = DiceCELoss(
        include_background=True, reduction="mean", to_onehot_y=True, softmax=True
    )
    model = model.to(torch.float32).to(device)
    model = DistributedDataParallel(model,device_ids=[device])
    #cofigs regarding model params 
    lr = conf["learn_rate"]
    momentum = conf["momentum"]
    batch_size = conf["batch_size"]
    cache_dir = conf["cache_dir"]
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=conf["epochs"],
    )  
    max_epochs = conf['epochs']
    m_rank = dist.get_rank() 
    best_metric =0 
    best_metric_epoch =0 
    if m_rank ==0: 
        log_path = help_utils.figure_version(
            conf["log_dir"]
        )  # TODO think about how you could perhaps continue training
        weight_path = conf["log_dir"].replace("model_logs", "model_checkpoints")
        if not os.path.isdir(weight_path):
            os.makedirs(weight_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        writer =  SummaryWriter(log_dir=log_path)
    else: 
        writer = None  #TODO make sure code works with NOne writeter
    #tr_dset.start()  #TODO this is only if i use the smart cache dataset 
    img_k = conf["img_key_name"]
    lbl_k = conf["lbl_key_name"]
    global_step_count =0 
    for epoch in range (max_epochs): 
        model.train()
        epoch_loss = 0
        step = 0
        for batch_n,batch_data in enumerate(train_dl): 
            if rank==0: 
                print(f"{rank} is on batch: {batch_n}",end='\r')
            inputs, labels = (batch_data[img_k], batch_data[lbl_k])
            if step == 0 and epoch % 2 == 0 and m_rank==0:
                help_utils.write_batches(
                    writer=writer,
                    inputs=inputs.detach(),
                    labels=labels.detach(),
                    epoch=epoch,
                    dset='train',
                    config=conf
                )
            optimizer.zero_grad()
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            local_loss = reduce_tensors(tensor=loss,world_size=dist.get_world_size()).cpu().item()
            if writer: 
                writer.add_scalar(
                    "batch_f_loss",
                    local_loss,
                    global_step=global_step_count,
                )
            global_step_count += 1 
        epoch_loss /= step
        lr_scheduler.step()
        if writer: 
            writer.add_scalar("epoch_loss", epoch_loss, global_step=epoch)
        all_d,all_l = eval_loop(model,val_dl,writer,epoch,device=device,config=conf) 
        dice_reduc = reduce_tensors(all_d,world_size=dist.get_world_size())
        loss_reduc = reduce_tensors(all_l,world_size=dist.get_world_size()) 
        if writer: 
            writer.add_scalar("val_loss",loss_reduc,global_step=epoch) 
            writer.add_scalar("val_dice",dice_reduc,global_step=epoch)
        if all_d > best_metric and rank==0:
            best_metric = all_d
            best_metric_epoch = epoch + 1
            torch.save(
                {
                    "conf": conf,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(weight_path, "best_metric_model.pth"),
            )
            print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {all_d:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
        if rank==0 and (epoch - best_metric_epoch) > 100 and abs(best_metric - all_d) < 0.01:
            #  we will exit training  because the model has not made large useful progres sin a reasonable amount of time
            print(f"Killign training due to insufficient progress")
            dist.destroy_process_group()
            break 
    dist.destroy_process_group()




def setup_repro(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(conf_in, trial=None):
    if trial:
        conf = optuna_gen(copy(conf_in), trial)
    else:
        conf = conf_in
    rank = conf_in['local_rank']
    seed = conf_in['seed']
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
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    _parse()
