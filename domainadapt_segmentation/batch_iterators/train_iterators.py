import os
import sys
import torch
import numpy as np
from ..helper_utils import utils as help_utils
from monai.transforms import AsDiscrete, Compose,Activations
from monai.losses import DiceLoss,DiceCELoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pdb 
from torch.nn.parallel import DistributedDataParallel 
import torch.distributed as dist 
import numpy as np 
from torch.nn import CosineEmbeddingLoss 
from ..helper_utils.utils import _proc3dbatch


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
    metric = DiceMetric(include_background=False,reduction="mean")
    model.eval()
    loss_function = DiceLoss(include_background=False,reduction="mean",to_onehot_y=True,sigmoid=True)
    all_losses = list()
    dice_scores = list()
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    post_label = Compose([Activations(to_onehot=2)])
    _step = 0 
    rank = dist.get_rank()  if len(config['device'])>=2 else 0
    print(f'{rank}: On Validation')
    #set up some inference terms 
    roi_size = config['spacing_vox_dim']
    batch_size = config['batch_size']
    with torch.no_grad():
        for i,val_data in enumerate(loader):
            val_inputs, val_labels = (
                val_data[img_k].to(device),
                val_data[lbl_k],
            ) 
            #this distinciton is needed because my 2D models need a way to compress the 2d patches to be (h,w) instead of (h,w,1).TODO: can i clean htat up?
            if config['2Dvs3D'] == "2D": 
                val_outputs= sliding_window_inference(inputs=val_inputs,roi_size=roi_size,sw_batch_size=batch_size,predictor=model,slide_window_compress=True,sw_device=device)
            else: 
                val_outputs= sliding_window_inference(inputs=val_inputs,roi_size=roi_size,sw_batch_size=batch_size,predictor=model,sw_device=device,mode='constant',device='cpu').to(device)
            loss = loss_function(val_outputs.to('cpu'),val_labels)
            val_outputs = [post_pred(i).to('cpu') for i in decollate_batch(val_outputs)]
            val_labels =  [post_label(i).to('cpu') for i in decollate_batch(val_labels)]
            if rank==0 and _step==0: 
                help_utils.write_pred_batches(
                        writer=writer,
                        inputs=val_data[img_k].detach(),
                        labels=val_labels,
                        preds=val_outputs,
                        epoch=epoch,
                        dset='val',
                        config=config, 
                        is_eval=True
                    ) 
            metric(y_pred=val_outputs, y=val_labels)
            metric_val= metric.aggregate("mean_batch")
            dice_scores.extend(metric_val)
            all_losses.append(loss)
            _step +=1 
        all_l = torch.mean(torch.stack(all_losses)).to(device)
        all_d = torch.mean(torch.vstack(dice_scores)).to(device)
        print(f"{rank} got avg dice of {all_d} and dice loss {all_l}")
    return all_d, all_l

def train_consistency(model=None,train_dl=None,optis=None,criterions=None,writer=None,global_step_count=None,epoch=None,conf=None):
    #TODO: add this as a loss function loss_func = CosineEmbeddingLoss(0.5)
    #i should perhaps match my items. 
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
        outputs,embeds = model(inputs) 
        loss = criterions['task'](outputs,labels)
        neg_far,po_close = criterions['consistency'](labels,embeds,device)
        const_loss = 0.25*neg_far + 0.75*po_close
        total_loss = loss + const_loss
        total_loss.backward()
        optis['task'].step() 
        epoch_loss += total_loss.cpu().detach()
        dice_loss = reduce_tensors(tensor=loss,world_size=dist.get_world_size()).cpu().item() if  world_size >=2 else loss
        consistency_loss = reduce_tensors(tensor=const_loss,world_size=dist.get_world_size()).cpu().item() if  world_size >=2 else loss
        ov_loss = reduce_tensors(tensor=total_loss,world_size=dist.get_world_size()).cpu().item() if world_size>=2 else loss 
        if writer: 
            writer.add_scalar(
                "batch_ov_loss",
                ov_loss,
                global_step=global_step_count,
            )
            writer.add_scalar(
                "batch_dice_loss",
                dice_loss,
                global_step=global_step_count,
            )
            writer.add_scalar(
                "batch_consistency_loss",
                consistency_loss,
                global_step=global_step_count,
            )
        global_step_count += 1
    epoch_loss /= step 
    epoch_loss =  epoch_loss.to(device)
    print(f" I am rank {rank} i have completed {global_step_count}")
    return epoch_loss,global_step_count

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

def train_basic_vae(model=None,train_dl=None,optis=None,criterions=None,writer=None,global_step_count=None,epoch=None,conf=None):
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
        outputs,recon_loss = model(inputs) 
        loss = criterions['task'](outputs,labels) + recon_loss
        loss.backward() 
        optis['task'].step() 
        epoch_loss += loss.cpu().detach()
        local_loss = reduce_tensors(tensor=loss,world_size=dist.get_world_size()).cpu().item() if  world_size >=2 else loss
        local_rec_loss = reduce_tensors(tensor=recon_loss,world_size=dist.get_world_size()).cpu().item() if  world_size >=2 else loss
        if writer: 
            writer.add_scalar(
                "batch_f_loss",
                local_loss,
                global_step=global_step_count,
            )
            writer.add_scalar(
                "batch_recon_loss",
                local_rec_loss,
                global_step=global_step_count,
            )
        global_step_count += 1
    if rank==0:
        view_recons(model,inputs,labels,writter=writer,epoch=epoch)
    epoch_loss /= step 
    epoch_loss =  epoch_loss.to(device)
    print(f" I am rank {rank} i have completed {global_step_count}")
    return epoch_loss,global_step_count
         
def view_recons(model,imgs,labels,writter,epoch):
    recon = model.module.get_recon(imgs)
    out_imgs,out_lbls = _proc3dbatch(recon,labels)
    if writter: 
        writter.add_images(f"train_recon",out_imgs,global_step=epoch)



        

