import os
import sys
import torch
import numpy as np
from ..helper_utils import utils as help_utils
from monai.transforms import AsDiscrete, Compose,Activations
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb 
from torch.nn.parallel import DistributedDataParallel 
import torch.distributed as dist 


def reduce_tensors(tensor,op=dist.ReduceOp.SUM,world_size=2): 
    tensor = tensor.clone() 
    dist.all_reduce(tensor,op) 
    tensor.div(world_size) 
    return tensor

def train_batch(
    model, loaders, optimizer, lr_scheduler, loss_function, device="cpu", config=None
):
    conf = config
    img_k = conf["img_key_name"]
    lbl_k = conf["lbl_key_name"]
    (
        train_loader,
        val_loader,
        _,
    ) = loaders  # prep my dataset loaders. during this phase we do not use the test_loader ever
    max_epochs = config["epochs"]
    best_metric = -10000000  # default value  that needed initialization
    global_step_count = 0  # TODO: resumable training would require a rewind of the clock. i.e aware of epoch and steps
    log_path = help_utils.figure_version(
        conf["log_dir"]
    )  # TODO think about how you could perhaps continue training
    weight_path = conf["log_dir"].replace("model_logs", "model_checkpoints")
    if not os.path.isdir(weight_path):
        os.makedirs(weight_path)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    for epoch in range(max_epochs):
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        print("activating the data loader")
        for batch_data in tqdm(train_loader, total=len(train_loader)):
            inputs, labels = (batch_data[img_k], batch_data[lbl_k])
            if step == 0 and epoch % 2 == 0:
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
            writer.add_scalar(
                "batch_f_loss",
                loss.cpu().detach().item(),
                global_step=global_step_count,
            )
            global_step_count += 1
        epoch_loss /= step
        lr_scheduler.step()
        writer.add_scalar("epoch_loss", epoch_loss, global_step=epoch)
        model.eval()
        val_metric, val_l = eval_loop(model, val_loader, writer, epoch, "val", config)
        if val_metric > best_metric:
            best_metric = val_metric
            best_metric_epoch = epoch + 1
            torch.save(
                {
                    "conf": config,
                    "state_dict": model.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(weight_path, "best_metric_model.pth"),
            )
            print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {val_metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
        if (epoch - best_metric_epoch) > 100 and abs(best_metric - val_metric) < 0.01:
            #  we will exit training  because the model has not made large useful progres sin a reasonable amount of time
            print(f"Killign training due to insufficient progress")
            sys.exit()


def eval_loop(model, loader, writer, epoch, device, config):
    roi_size = config["spacing_vox_dim"]
    img_k = config["img_key_name"]
    lbl_k = config["lbl_key_name"]
    num_seg_labels = config["num_seg_labels"]
    metric = DiceMetric(include_background=True,reduction="mean")
    model.eval()
    loss_function = DiceLoss(include_background=True, reduction="mean",to_onehot_y=False,softmax=False)
    all_losses = list()
    dice_scores = list()
    to_probs = torch.nn.Softmax(dim=0)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    _step = 0 
    rank = dist.get_rank()  if len(config['device'])>=2 else 0
    print(f'{rank}: On Validation')
    #set up some inference terms 
    roi_size = config['spacing_vox_dim']
    batch_size = config['batch_size']
    with torch.no_grad():
        for i,val_data in enumerate(loader):
            if rank==0 and _step==0 and epoch==0: 
                help_utils.write_batches(
                    writer=writer,
                    inputs=val_data[img_k].detach(),
                    labels=val_data[lbl_k].detach(),
                    epoch=epoch,
                    dset='val',
                    config=config, 
                    is_eval=True
                ) 

            val_inputs, val_labels = (
                val_data[img_k].to(device),
                val_data[lbl_k].to(device),
            ) 
            #this distinciton is needed because my 2D models need a way to compress the 2d patches to be (h,w) instead of (h,w,1).TODO: can i clean htat up?
            if config['2Dvs3D'] == "2D": 
                val_outputs= sliding_window_inference(inputs=val_inputs,roi_size=roi_size,sw_batch_size=batch_size,predictor=model,slide_window_compress=True,sw_device=device)
            else: 
                val_outputs= sliding_window_inference(inputs=val_inputs,roi_size=roi_size,sw_batch_size=batch_size,predictor=model,sw_device=device,mode='constant',device='cpu').to(device)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels =  decollate_batch(val_labels)
            metric_val = metric(y_pred=val_outputs, y=val_labels)
            metric.reset()
            dice_scores.append(metric_val)
            print(metric_val.shape)
            loss = sum(
                loss_function(v_o, v_l) for v_o, v_l in zip(val_outputs, val_labels)
            )
            all_losses.append(loss)
        all_l = torch.mean(torch.stack(all_losses)).to(device)
        all_d = torch.mean(torch.vstack(dice_scores)).to(device)
        print(f"{rank} got avg dice of {all_d} and dice loss {all_l}")
    return all_d, all_l


def train_basic(model=None,train_dl=None,optis=None,criterions=None,writer=None,global_step_count=None,epoch=None,conf=None):
    img_k = conf['img_key_name'] 
    lbl_k = conf['lbl_key_name']
    model.train()
    rank = dist.get_rank() if len(conf['device'])>=2  else 0 
    world_size = dist.get_world_size() if len(conf['device'])>=2 else 1
    step= 0  
    device = torch.device(f"cuda:{rank}")
    epoch_loss = 0  
    for batch_n,batch_data in enumerate(train_dl): 
        print(f"{rank} is on batch: {batch_n} using GPU {device}",end='\r') 
        inputs, labels = (batch_data[img_k], batch_data[lbl_k])
        if step == 0 and epoch % 2 == 0 and rank==0:
            help_utils.write_batches(
                writer=writer,
                inputs=inputs.detach(),
                labels=labels.detach(),
                epoch=epoch,
                dset='train',
                config=conf
            ) 
        for e in optis: 
            optis[e].zero_grad() 
        step +=1 
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs) 
        loss = criterions['task'](outputs,labels)
        loss.backward() 
        optis['task'].step() 
        epoch_loss += loss.cpu().detach()
        local_loss = reduce_tensors(tensor=loss,world_size=dist.get_world_size()).cpu().item() if  world_size >=2 else loss
        if writer: 
            writer.add_scalar(
                "batch_f_loss",
                local_loss,
                global_step=global_step_count,
            )
        global_step_count += 1
    epoch_loss /= step 
    epoch_loss =  epoch_loss.to(device)
    print(f" I am rank {rank} i have completed {global_step_count}")
    return epoch_loss,global_step_count
         



        

