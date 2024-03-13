import torch
import os
import random
import os
from   .helper_utils import configs as help_configs
from  .helper_utils import data_io as help_io
from .helper_utils import transforms as help_transforms
from .helper_utils import utils as help_utils
from .batch_iterators.train_iterators import *
from .data_factories.kits_factory import kit_factory
from monai.data import SmartCacheDataset,partition_dataset
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

#selecting optimizer 
#TODO: perhpas this should go elsewhere 
def  get_params_list(model,name_select): 
    param_list = list() 
    if name_select=='all':
        return list(model.parameters() )
    if name_select=='domain': 
        param_list = list()
        for n,e in model.named_parameters():
            if n.startswith('discrim'):
                param_list.append(e)
        return param_list
    if name_select=='unet': 
        for n,e in model.named_parameters(): 
            if not n.startswith('discrim'):
                param_list.append(e)
        return param_list
def build_optimizers(model,conf=None): 
    lr = conf["learn_rate"]
    momentum = conf["momentum"]
    if conf['train_mode']=='vanilla' or conf['train_mode']=='vae': 
        optis = dict() 
        optis['task'] = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    if conf['train_mode']=='dinsdale':
        optim_seg = optim.Adam(get_params_list(model,'unet'),lr=0.1)
        opti_dm = optim.Adam(get_params_list(model,'domain'),lr=0.1) 
        opti_conf = optim.Adam(get_params_list(model,'all'),lr=0.1)
        optis = {'task':optim_seg,'domain_optim':opti_dm,'confuse_opti':opti_conf}
    if conf['train_mode']=='consistency': 
        optim_all = optim.Adam(get_params_list(model,'all'),lr=0.1) 
        optis = {'task':optim_all}
    if conf['train_mode']=='debias_two_branch':
        optis = dict()  #TODO: do i just have 1 parameter or do i have a tone of parameters for this 
        optis['task'] = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
        optis['domain']= optim.SGD(filter_params(model,'bottleneck_branch') + filter_params(model,'mask_branch'),lr=0.1)  
        optis['confusion'] = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    if conf['train_mode']=='debias_one_branch':
        optis = dict()  #TODO: do i just have 1 parameter or do i have a tone of parameters for this 
        optis['task'] = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
        print("filtering weights for bottleneck")
        optis['domain']= optim.SGD(filter_params(model,'bottleneck_branch'),lr=0.1)  
        optis['confusion'] = optim.SGD(model.parameters(),lr=lr,momentum=momentum)
    return optis  
def filter_params(model,param_pattern): 
    param_list = list() 
    for n,e in model.named_parameters():
        if param_pattern in n: 
            print(n)
            param_list.append(e)
    return param_list


def my_consistency_loss(batch_lbl,flat_vec,device):
    instances = (batch_lbl.sum(axis=-1).sum(axis=-1).sum(axis=-1)>=1).to(torch.int).cpu()
    num_pos = instances.sum()
    num_ne = num_pos 
    ne_idx =  torch.where(instances==0)[0]
    pos_idx = torch.where(instances==1)[0]
    neg_idx_sample = torch.tensor(np.random.choice(ne_idx,num_ne)).to(device)
    neg_batch = flat_vec[neg_idx_sample]
    pos_batch = flat_vec[pos_idx]
    loss_func = CosineEmbeddingLoss(0.5)
    neg_labels = -1*torch.ones((num_ne,)).to(device)
    make_pos_neg_far = loss_func(pos_batch,neg_batch,neg_labels)
    rand_per = torch.tensor(np.random.permutation(torch.tensor(range(pos_batch.shape[0])))).to(device)
    pos_labels = torch.ones((num_pos,)).to(device)
    make_pos_pos_close = loss_func(pos_batch,pos_batch[rand_per],pos_labels) 
    return make_pos_neg_far , make_pos_pos_close
    


def build_criterions(conf=None): 
    if conf['train_mode']=='vanilla' or conf['train_mode']=='vae': 
        criterions = dict() 
        criterions['task'] = DiceCELoss(include_background=True,reduction="mean",to_onehot_y=True,sigmoid=True)
    if conf['train_mode']=='dinsdale': 
        criterions = dict() 
        criterions['task'] = DiceCELoss(include_background=True,reduction="mean",to_onehot_y=True,sigmoid=True)
        criterions['domain'] = torch.nn.CrossEntropyLoss()
        criterions['conf'] = confusion_loss()
    if conf['train_mode']=='consistency': 
        criterions = dict() 
        criterions['task'] = DiceCELoss(include_background=True,reduction="mean",to_onehot_y=True,sigmoid=True)
        criterions['consistency'] = my_consistency_loss
    if conf['train_mode']=='debias_two_branch' or conf['train_mode']=='debias_one_branch':
        criterions = dict() 
        criterions['task'] = DiceCELoss(include_background=True,reduction="mean",to_onehot_y=True,sigmoid=True)
        criterions['domain'] = torch.nn.CrossEntropyLoss(reduction="none")
        criterions['conf'] = confusion_loss(reduction='none')
    return criterions

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
        for e in gpus: 
            print(e.split(':'))
        device_nums = [e.split(':')[1] for e in gpus]
        os.environ['CUDA_VISIBLE_DEVICES']= ",".join(device_nums) 
        world_size = len(gpus)
        if world_size==1: 
            print('we are not doing parallel')
            dummy_main(0,1,conf) 
        else:
            mp.spawn(dummy_main,args=(world_size,conf),nprocs=world_size,join=True,)
        #main(conf)

def train_dispatch(model=None,train_dl=None,optis=None,
                    criterions=None,writer=None,global_step_count=None,
                    epoch=None,conf=None):
    train_mode= conf['train_mode']
    if train_mode =='consistency': 
        epoch_loss, global_step_count = train_consistency(model=model,train_dl=train_dl,optis=optis,
                    criterions=criterions,writer=writer,global_step_count=global_step_count,
                    epoch=epoch,conf=conf)
    if train_mode == 'vanilla': 
        epoch_loss, global_step_count = train_basic(model=model,train_dl=train_dl,optis=optis,
                    criterions=criterions,writer=writer,global_step_count=global_step_count,
                    epoch=epoch,conf=conf)
    if train_mode == 'vae': 
        epoch_loss, global_step_count = train_basic_vae(model=model,train_dl=train_dl,optis=optis,
                    criterions=criterions,writer=writer,global_step_count=global_step_count,
                    epoch=epoch,conf=conf)
    if train_mode =='debias_two_branch': 
        epoch_loss, global_step_count = train_two_branch(model=model,train_dl=train_dl,optis=optis,
                    criterions=criterions,writer=writer,global_step_count=global_step_count,
                    epoch=epoch,conf=conf)
    if train_mode =='debias_one_branch': 
        epoch_loss, global_step_count = train_one_branch(model=model,train_dl=train_dl,optis=optis,
                    criterions=criterions,writer=writer,global_step_count=global_step_count,
                    epoch=epoch,conf=conf)
    return epoch_loss,global_step_count

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
    if world_size >=2:
        dist.init_process_group(backend='nccl',rank=rank,world_size=world_size,init_method="env://") 
        m_rank = dist.get_rank() 
    else: 
        m_rank = 0 
    #here we set up the partitioning of the training data across all the workers 
    ##SETUP ALL THE DATASET RELATED ITEMS 
    train, val, test = help_io.load_data(conf["data_path"]) # this is just a list of dictionaries 
    # use short circuitting to check if dev is  a field
    if "dev" in conf.keys() and conf["dev"] == True:
        print(
            "we are outputting to devset we are therefore using a smaller train sample for dev"
        )
        train = random.sample(train, 20)
        val = random.sample(val, 20)
    train_transform, val_transform = help_transforms.gen_transforms(conf) # no changes needed for default transforms 
    if world_size >=2: 
        tr_part = partition_dataset(train,num_partitions=dist.get_world_size(),shuffle=False,even_divisible=False) [dist.get_rank()] # this will create a dataset  for each partion 
        val_part = partition_dataset(val,num_partitions=dist.get_world_size(),shuffle=False,even_divisible=False)[dist.get_rank()] 
    else: 
        tr_part  = train 
        val_part = val  
    dset = kit_factory('cached')
    batch_size = conf['batch_size']
    cache_dir = conf['cache_dir']
    tr_dset = dset(tr_part,transform= train_transform,cache_dir=cache_dir)
    val_dset = dset(val_part,transform=val_transform,cache_dir=cache_dir)
    train_mode = conf['train_mode']
    if train_mode=='debias_two_branch' or train_mode=='debias_one_branch':
        print("adding two branch")
        sampler = makeWeightedsampler(tr_part)
        shuffle=False
    else: 
        sampler =None
        shuffle=True
    num_workers = conf['num_workers']
    train_dl = DataLoader(
        tr_dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=help_transforms.ramonPad()
    )
    val_dl = DataLoader(
        val_dset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=help_transforms.ramonPad()
    )
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device) 
    #load the model 
    model = model_factory(config=conf)
    loss_function = DiceCELoss(
        include_background=True, reduction="mean", to_onehot_y=True, softmax=True
    )
    model = model.to(torch.float32).to(device)
    #model = torch.compile(model,fullgraph=False,dynamic=True)
    if world_size >=2: 
        model = DistributedDataParallel(model,device_ids=[device])
    #cofigs regarding model params, 
    batch_size = conf["batch_size"]
    cache_dir = conf["cache_dir"]
    optis = build_optimizers(model,conf=conf) 
    criterions = build_criterions(conf) 
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optis['task'],
        total_iters=conf["epochs"],
        power=1.5
    )  
    max_epochs = conf['epochs']
    best_metric =0 
    best_metric_epoch =0 
    if m_rank ==0: 
        log_path = help_utils.figure_version(
            conf["log_dir"]
        )  # TODO think about how you could perhaps continue training
        weight_path =log_path 
        if not os.path.isdir(weight_path):
            os.makedirs(weight_path)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        writer =  SummaryWriter(log_dir=log_path)
    else: 
        writer = None  #TODO make sure code works with NOne writeter
    global_step_count =0  
    all_d = 0 
    for epoch in range (max_epochs): 
        epoch_loss, global_step_count = train_dispatch(model=model,train_dl=train_dl,optis=optis,
                    criterions=criterions,writer=writer,global_step_count=global_step_count,
                    epoch=epoch,conf=conf)
        lr_scheduler.step()
        epoch_loss = reduce_tensors(epoch_loss,world_size=dist.get_world_size()).item() if world_size >=2 else epoch_loss.item()
        print(f"{rank} epoch loss: {epoch_loss}")
        if writer: 
            writer.add_scalar("epoch_loss", epoch_loss, global_step=epoch) 
        if (epoch and epoch %10 ==0 )or conf['dev']==True:
            all_d,all_l = eval_loop(model,val_dl,writer,epoch,device=device,config=conf) 
            print("Done with the  eval loop")
            dice_reduc = reduce_tensors(all_d,world_size=dist.get_world_size()) if world_size >=2 else all_d
            loss_reduc = reduce_tensors(all_l,world_size=dist.get_world_size()) if world_size >=2 else all_d
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
            if world_size >=2: 
                print(f"Killign training due to insufficient progress")
                dist.destroy_process_group()
            break
    if world_size>=2:  
        dist.destroy_process_group()




def setup_repro(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)




if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    _parse()
