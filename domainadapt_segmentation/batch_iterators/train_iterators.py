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
import pdb 
from torch.nn.utils import clip_grad_norm
from copy import deepcopy 

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
    phase_k = conf["phase"] #TODO:Make this part of the ocnfirugation and an expected
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
        model.train()
        epoch_loss = 0
        step = 0
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
            mask_out,embed_pred,mask_pred = model(inputs)
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
    metric = DiceMetric(include_background=False,reduction="mean_batch")
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
            metric_val= metric.aggregate(reduction="mean_batch")
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
    device = torch.device(conf['device'][rank])
    epoch_loss = 0  
    for batch_n,batch_data in enumerate(train_dl): 
        inputs, labels = (deepcopy(batch_data[img_k]), deepcopy(batch_data[lbl_k]) )
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

def train_two_branch(model=None,train_dl=None,optis=None,criterions=None,writer=None,global_step_count=None,epoch=None,conf=None):
    img_k = conf['img_key_name'] 
    lbl_k = conf['lbl_key_name']
    phase_k = 'phase'
    model.train()
    rank = dist.get_rank() if len(conf['device'])>=2  else 0 
    world_size = dist.get_world_size() if len(conf['device'])>=2 else 1
    step= 0  
    device = torch.device(f"cuda:{rank}")
    epoch_loss = 0  
    if epoch <200:  #TODO: make this a configuration item
        confusion_lambda = 0 
    else: 
        confusion_lambda = 0.1
    for batch_n,batch_data in enumerate(train_dl): 
        print(f"{rank} is on batch: {batch_n} using GPU {device}",end='\r') 
        torch.cuda.empty_cache()
        inputs, labels,phase = (batch_data[img_k], batch_data[lbl_k],batch_data[phase_k])
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
        has_kidney = (labels.sum(1).sum(1).sum(1).sum(1) !=0).to(torch.int)
        phase = phase.to(device) 
        #train the segmentation. entire unet structure
        seg_pred,embed_domain_pred,mask_domain_pred= model(inputs) 
        seg_loss = criterions['task'](seg_pred,labels)
        embed_domain_loss = (criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
        mask_domain_loss = (criterions['domain'](mask_domain_pred,phase)*has_kidney).mean()
        total_loss = seg_loss + 0.0*embed_domain_loss + 0.0*mask_domain_loss  #we need the loss to be included in calculation for DDP but grad can be zero
        total_loss.backward() 
        optis['task'].step()   
        #train the discriminators.
        seg_pred,embed_domain_pred,mask_domain_pred= model(inputs) 
        seg_loss = criterions['task'](seg_pred,labels)
        embed_domain_loss = (criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
        mask_domain_loss = (criterions['domain'](mask_domain_pred,phase)*has_kidney).mean()
        domain_loss =  embed_domain_loss + mask_domain_loss 
        total_loss =  0.0*seg_loss + domain_loss  #ddp based clipping. if not included you will suffer
        total_loss.backward()
        clip_grad_norm(model.parameters(),max_norm=2.0,norm_type=2.0)
        optis['domain'].step() 
        for e in optis: 
            optis[e].zero_grad() 
        #penalize the  entire model. with seg loss and confusion loss
        seg_pred,embed_domain_pred,mask_domain_pred= model(inputs) 
        embed_conf_loss = torch.mean(criterions['conf'](embed_domain_pred,phase)*has_kidney)
        mask_conf_loss = torch.mean(criterions['conf'](mask_domain_pred,phase)*has_kidney)
        seg_loss = criterions['task'](seg_pred,labels)
        conf_loss = confusion_lambda*( embed_conf_loss + mask_conf_loss ) 
        new_conf_loss = conf_loss + seg_loss
        new_conf_loss.backward() 
        clip_grad_norm(model.parameters(),max_norm=2.0,norm_type=2.0)
        optis['confusion'].step()
        for e in optis: 
            optis[e].zero_grad() 
        #aggrgate all the losses for metrics
        total_loss =  seg_loss.detach() + domain_loss.detach() + conf_loss.detach()
        #if torch.isnan(total_loss).any():
        #    pdb.set_trace()
        epoch_loss += total_loss
        local_loss = reduce_metrics(epoch_loss,world_size=world_size)
        local_dice_loss = reduce_metrics(seg_loss,world_size=world_size)
        local_domain_embed_loss = reduce_metrics(embed_domain_loss,world_size=world_size)
        local_domain_mask_loss = reduce_metrics(mask_domain_loss,world_size=world_size)
        local_conf_embed_loss = reduce_metrics(embed_conf_loss,world_size=world_size)
        local_conf_mask_loss = reduce_metrics(mask_conf_loss,world_size=world_size)
        if writer: 
            writer.add_scalar("batch_DICE+CE_loss",local_dice_loss,global_step=global_step_count)
            writer.add_scalar("batch_total_loss",local_loss,global_step=global_step_count)
            writer.add_scalar("embed_domain_loss",local_domain_embed_loss,global_step=global_step_count)
            writer.add_scalar("mask_domain_loss",local_domain_mask_loss,global_step=global_step_count) 
            writer.add_scalar("embed_confusion_loss",local_conf_embed_loss,global_step=global_step_count)
            writer.add_scalar("mask_confusion_loss",local_conf_mask_loss,global_step=global_step_count)
        global_step_count += 1
    epoch_loss /= step 
    epoch_loss =  epoch_loss.to(device)
    print(f" I am rank {rank} i have completed {global_step_count}")
    return epoch_loss,global_step_count
def train_one_branch(model=None,train_dl=None,optis=None,criterions=None,writer=None,global_step_count=None,epoch=None,conf=None):
    img_k = conf['img_key_name'] 
    lbl_k = conf['lbl_key_name']
    phase_k = 'phase'
    model.train()
    rank = dist.get_rank() if len(conf['device'])>=2  else 0 
    world_size = dist.get_world_size() if len(conf['device'])>=2 else 1
    step= 0  
    device = torch.device(f"cuda:{rank}")
    epoch_loss = 0  
    if epoch <200: 
        confusion_lambda = 0 
    else: 
        confusion_lambda = 0.1
    for batch_n,batch_data in enumerate(train_dl): 
        print(f"{rank} is on batch: {batch_n} using GPU {device}",end='\r') 
        torch.cuda.empty_cache()
        inputs, labels,phase = (batch_data[img_k], batch_data[lbl_k],batch_data[phase_k])
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
        has_kidney = (labels.sum(1).sum(1).sum(1).sum(1) !=0).to(torch.int)
        phase = phase.to(device) 
        #train the segmentation. entire unet structure
        seg_pred,embed_domain_pred= model(inputs) 
        seg_loss = criterions['task'](seg_pred,labels)
        embed_domain_loss = (criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
        total_loss =  seg_loss + 0.0*embed_domain_loss #we need to do this so dataParallel doesn't yell at me
        total_loss.backward()
        clip_grad_norm(model.parameters(),max_norm=2.0,norm_type=2.0)
        optis['task'].step()   
        for e in optis: 
            optis[e].zero_grad() 
        #train the discriminator 
        seg_pred,embed_domain_pred= model(inputs) 
        seg_loss = criterions['task'](seg_pred,labels)
        embed_domain_loss = (criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
        clip_grad_norm(model.parameters(),max_norm=2.0,norm_type=2.0)
        total_loss = 0.0*seg_loss + embed_domain_loss #block out the seg loss for ddp calcs
        total_loss.backward()
        optis['domain'].step() 
        for e in optis: 
            optis[e].zero_grad() 
        #penalize the discriminators only  with confusion loss
        seg_pred,embed_domain_pred = model(inputs) 
        embed_conf_loss = torch.mean(criterions['conf'](embed_domain_pred,phase)*has_kidney)
        seg_loss = criterions['task'](seg_pred,labels)
        conf_loss = ( embed_conf_loss ) 
        new_conf_loss = confusion_lambda*(conf_loss + seg_loss)
        new_conf_loss.backward()   
        clip_grad_norm(model.parameters(),max_norm=2.0,norm_type=2.0)
        optis['confusion'].step()
        for e in optis: 
            optis[e].zero_grad() 
        #aggrgate all the losses for metrics
        total_loss =  seg_loss.detach() + embed_domain_loss.detach() + conf_loss.detach()
        if torch.isnan(total_loss).any():
            pdb.set_trace()
        epoch_loss += total_loss

        local_loss = reduce_metrics(epoch_loss,world_size=world_size)
        local_dice_loss = reduce_metrics(seg_loss,world_size=world_size)
        local_domain_embed_loss = reduce_metrics(embed_domain_loss,world_size=world_size)
        local_conf_embed_loss = reduce_metrics(embed_conf_loss,world_size=world_size)
        if writer: 
            writer.add_scalar("batch_DICE+CE_loss",local_dice_loss,global_step=global_step_count)
            writer.add_scalar("batch_total_loss",local_loss,global_step=global_step_count)
            writer.add_scalar("embed_domain_loss",local_domain_embed_loss,global_step=global_step_count)
            writer.add_scalar("embed_confusion_loss",local_conf_embed_loss,global_step=global_step_count)
        global_step_count += 1
    epoch_loss /= step 
    epoch_loss =  epoch_loss.to(device)
    print(f" I am rank {rank} i have completed {global_step_count}")
    return epoch_loss,global_step_count
def train_one_branch_adv(model=None,train_dl=None,optis=None,criterions=None,writer=None,global_step_count=None,epoch=None,conf=None):
    img_k = conf['img_key_name'] 
    lbl_k = conf['lbl_key_name']
    phase_k = 'phase'
    model.train()
    rank = dist.get_rank() if len(conf['device'])>=2  else 0 
    world_size = dist.get_world_size() if len(conf['device'])>=2 else 1
    step= 0  
    device = torch.device(f"cuda:{rank}")
    epoch_loss = 0  
    if epoch <200: 
        confusion_lambda = 0 
    else: 
        confusion_lambda = 0.1
    for batch_n,batch_data in enumerate(train_dl): 
        print(f"{rank} is on batch: {batch_n} using GPU {device}",end='\r') 
        torch.cuda.empty_cache()
        inputs, labels,phase = (batch_data[img_k], batch_data[lbl_k],batch_data[phase_k])
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
        has_kidney = (labels.sum(1).sum(1).sum(1).sum(1) !=0).to(torch.int)
        phase = phase.to(device) 
        #train the segmentation. entire unet structure
        seg_pred,embed_domain_pred= model(inputs) 
        seg_loss = criterions['task'](seg_pred,labels)
        embed_domain_loss = (criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
        total_loss =  seg_loss + 0.0*embed_domain_loss #we need to do this so dataParallel doesn't yell at me
        total_loss.backward()
        clip_grad_norm(model.parameters(),max_norm=2.0,norm_type=2.0)
        optis['task'].step()   
        for e in optis: 
            optis[e].zero_grad() 
        #train the discriminator 
        model.debias = False
        seg_pred,embed_domain_pred= model(inputs) 
        seg_loss = criterions['task'](seg_pred,labels)
        embed_domain_loss = (criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
        clip_grad_norm(model.parameters(),max_norm=2.0,norm_type=2.0)
        total_loss = 0.0*seg_loss + embed_domain_loss #block out the seg loss for ddp calcs
        total_loss.backward()
        optis['domain'].step() 
        for e in optis: 
            optis[e].zero_grad() 
        #penalize the discriminators only  with confusion loss
        model.debias = True
        seg_pred,embed_domain_pred = model(inputs) 
        embed_conf_loss = torch.mean(criterions['domain'](embed_domain_pred,phase)*has_kidney)
        seg_loss = criterions['task'](seg_pred,labels)
        conf_loss = ( embed_conf_loss ) 
        new_conf_loss = confusion_lambda*(conf_loss + seg_loss)
        new_conf_loss.backward()   
        clip_grad_norm(model.parameters(),max_norm=2.0,norm_type=2.0)
        optis['all'].step()
        for e in optis: 
            optis[e].zero_grad() 
        #aggrgate all the losses for metrics
        total_loss =  seg_loss.detach() + embed_domain_loss.detach() + conf_loss.detach()
        if torch.isnan(total_loss).any():
            pdb.set_trace()
        epoch_loss += total_loss

        local_loss = reduce_metrics(epoch_loss,world_size=world_size)
        local_dice_loss = reduce_metrics(seg_loss,world_size=world_size)
        local_domain_embed_loss = reduce_metrics(embed_domain_loss,world_size=world_size)
        local_conf_embed_loss = reduce_metrics(embed_conf_loss,world_size=world_size)
        if writer: 
            writer.add_scalar("batch_DICE+CE_loss",local_dice_loss,global_step=global_step_count)
            writer.add_scalar("batch_total_loss",local_loss,global_step=global_step_count)
            writer.add_scalar("embed_domain_loss",local_domain_embed_loss,global_step=global_step_count)
            writer.add_scalar("embed_confusion_loss",local_conf_embed_loss,global_step=global_step_count)
        global_step_count += 1
    epoch_loss /= step 
    epoch_loss =  epoch_loss.to(device)
    print(f" I am rank {rank} i have completed {global_step_count}")
    return epoch_loss,global_step_count

def  reduce_metrics(tens,world_size): 
    out = reduce_tensors(tensor=tens,world_size=world_size) if world_size >= 2 else tens
    return out 
def view_recons(model,imgs,labels,writter,epoch):
    recon = model.module.get_recon(imgs)
    out_imgs,out_lbls = _proc3dbatch(recon,labels)
    if writter: 
        writter.add_images(f"train_recon",out_imgs,global_step=epoch)



        
def train_one_branch_dinsdale(model=None,train_dl=None,optis=None,criterions=None,writer=None,global_step_count=None,epoch=None,conf=None):
    img_k = conf['img_key_name'] 
    lbl_k = conf['lbl_key_name']
    phase_k = 'phase'
    model.train()
    rank = dist.get_rank() if len(conf['device'])>=2  else 0 
    world_size = dist.get_world_size() if len(conf['device'])>=2 else 1
    step= 0  
    device = torch.device(f"cuda:{rank}")
    epoch_loss = 0  
    if epoch <200: 
        confusion_lambda = 0 
    else: 
        confusion_lambda = 0.1
    for batch_n,batch_data in enumerate(train_dl): 
        print(f"{rank} is on batch: {batch_n} using GPU {device}",end='\r') 
        torch.cuda.empty_cache()
        inputs, labels,phase = (batch_data[img_k], batch_data[lbl_k],batch_data[phase_k])
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
        phase = phase.to(device) 
        #train the segmentation. entire unet structure
        seg_pred,embed_domain_pred= model(inputs) 
        seg_loss = criterions['task'](seg_pred,labels)
        embed_domain_loss = (criterions['domain'](embed_domain_pred,phase)).mean()
        total_loss =  seg_loss + 0.0*embed_domain_loss #we need to do this so dataParallel doesn't yell at me
        total_loss.backward()
        optis['task'].step()   
        for e in optis: 
            optis[e].zero_grad() 
        #train the discriminator 
        seg_pred,embed_domain_pred= model(inputs) 
        seg_loss = criterions['task'](seg_pred,labels)
        embed_domain_loss = (criterions['domain'](embed_domain_pred,phase)).mean()
        total_loss = 0.0*seg_loss + embed_domain_loss #block out the seg loss for ddp calcs
        total_loss.backward()
        optis['domain'].step() 
        for e in optis: 
            optis[e].zero_grad() 
        #penalize the discriminators only  with confusion loss
        seg_pred,embed_domain_pred = model(inputs) 
        embed_conf_loss = torch.mean(criterions['conf'](embed_domain_pred,phase))
        seg_loss = criterions['task'](seg_pred,labels)
        conf_loss = ( embed_conf_loss ) 
        new_conf_loss = confusion_lambda*(conf_loss + seg_loss)
        new_conf_loss.backward()   
        optis['confusion'].step()
        for e in optis: 
            optis[e].zero_grad() 
        #aggrgate all the losses for metrics
        total_loss =  seg_loss.detach() + embed_domain_loss.detach() + conf_loss.detach()
        if torch.isnan(total_loss).any():
            pdb.set_trace()
        epoch_loss += total_loss

        local_loss = reduce_metrics(epoch_loss,world_size=world_size)
        local_dice_loss = reduce_metrics(seg_loss,world_size=world_size)
        local_domain_embed_loss = reduce_metrics(embed_domain_loss,world_size=world_size)
        local_conf_embed_loss = reduce_metrics(embed_conf_loss,world_size=world_size)
        if writer: 
            writer.add_scalar("batch_DICE+CE_loss",local_dice_loss,global_step=global_step_count)
            writer.add_scalar("batch_total_loss",local_loss,global_step=global_step_count)
            writer.add_scalar("embed_domain_loss",local_domain_embed_loss,global_step=global_step_count)
            writer.add_scalar("embed_confusion_loss",local_conf_embed_loss,global_step=global_step_count)
        global_step_count += 1
    epoch_loss /= step 
    epoch_loss =  epoch_loss.to(device)
    print(f" I am rank {rank} i have completed {global_step_count}")
    return epoch_loss,global_step_count


def train_prototype(model=None,train_dl=None,optis=None,criterions=None,writer=None,global_step_count=None,epoch=None,conf=None):
    img_k = conf['img_key_name'] 
    lbl_k = conf['lbl_key_name']
    model.train() 
    prototype =  model.conv_final[2].conv.weight.view(2,8) 

    rank = dist.get_rank() if len(conf['device'])>=2  else 0 
    world_size = dist.get_world_size() if len(conf['device'])>=2 else 1
    step= 0  
    device = torch.device(f"cuda:{rank}")
    epoch_loss = 0  
    for batch_n,batch_data in enumerate(train_dl): 
        #print(f"{rank} is on batch: {batch_n} using GPU {device}") 
        inputs, labels = (deepcopy(batch_data[img_k]), deepcopy(batch_data[lbl_k]) )
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