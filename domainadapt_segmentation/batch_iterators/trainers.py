
from torch.utils.tensorboard import SummaryWriter 
from  torch import nn 
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch import optim 
from tqdm import tqdm 
import torch 
from monai.transforms import AsDiscrete, Compose,Activations
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from ..helper_utils.utils import write_pred_batches
import os 
from ..helper_utils.utils  import confusion_loss 
from torch.nn.utils import clip_grad_norm
import pdb 
from .trainer_factory import TrainerRegister 

@TrainerRegister.register("DiceTrainer")
class DiceTrainer(object):
    def __init__(self,model,device='cuda:0',tb_writter=None,conf=None,data_loaders=None):
        self.model=model
        self.tb:SummaryWriter= tb_writter 
        self.conf=conf 
        self.tr_dl=data_loaders['train']
        self.val_dl=data_loaders['val']
        self.ts_dl=data_loaders['test']
        self.total_epochs = conf['epochs']
        self.c_epoch = 0 
        self.device= device 
        self.gb_step=0
        self.img_k = conf['img_key_name'] 
        self.lbl_k = conf['lbl_key_name']
        self.init_optims()
        self._build_criterions()
    def _build_criterions(self): 
        self.criterions = dict() 
        self.criterions['dice'] = DiceCELoss(to_onehot_y=True)
        self.metrics = dict() 
        self.metrics['dice'] = DiceMetric(include_background=False,reduction="mean_batch")
    def _log_model_graph(self):
        self.model=  self.model.eval()
        vox_sample = torch.rand([1,1] +self.conf['spacing_vox_dim'])
        self.tb.add_graph(self.model,vox_sample.to(self.device))
    def init_optims(self):
        self.opti : optim.SGD = optim.SGD(self.model.parameters(),lr=self.conf['learn_rate'])
        self.sch = optim.lr_scheduler.PolynomialLR(self.opti,total_iters=self.total_epochs,power=1.1)
    def train_epoch(self):
        self.model = self.model.train() 
        for i,batch in tqdm(enumerate(self.tr_dl),total=len(self.tr_dl)):
            inputs, labels = (batch[self.img_k], batch[self.lbl_k])
            self.opti.zero_grad()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterions['dice'](outputs,labels)  
            loss.backward()
            self.opti.step()
            self.tb.add_scalar(
                "t_batch_f_loss",
                loss.cpu().detach().item(),
                global_step=self.gb_step,
            )
            self.opti.zero_grad()
            self.gb_step +=1 
        self.c_epoch +=1 
    def val_epoch(self):
        self.model=  self.model.eval()
        roi_size = self.conf["spacing_vox_dim"]
        num_seg_labels = self.conf["num_seg_labels"]
        metric = DiceMetric(include_background=False,reduction="mean_batch")
        self.model = self.model.eval()
        all_losses = list()
        dice_scores = list()
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        post_label = Compose([Activations(to_onehot=num_seg_labels)])
        _step = 0 
        batch_size = self.conf['batch_size']
        with torch.no_grad():
            for i,val_data in enumerate(self.val_dl):
                val_inputs, val_labels = (
                    val_data[self.img_k].to(self.device),
                    val_data[self.lbl_k],
                ) 
                #this distinciton is needed because my 2D models need a way to compress the 2d patches to be (h,w) instead of (h,w,1).TODO: can i clean htat up?
                val_outputs= sliding_window_inference(inputs=val_inputs,roi_size=roi_size,sw_batch_size=batch_size,predictor=self.model,sw_device=self.device,mode='constant',device='cpu').to(self.device)
                loss =  self.criterions['dice'](val_outputs.to('cpu'),val_labels)
            val_outputs = [post_pred(i).to('cpu') for i in decollate_batch(val_outputs)]
            val_labels =  [post_label(i).to('cpu') for i in decollate_batch(val_labels)]
            if _step==0: 
                write_pred_batches(
                        writer=self.tb,
                        inputs=val_data[self.img_k].detach(),
                        labels=val_labels,
                        preds=val_outputs,
                        epoch=self.epoch,
                        dset='val',
                        config=self.config, 
                        is_eval=True
                    ) 
            metric(y_pred=val_outputs, y=val_labels)
            metric_val= metric.aggregate(reduction="mean_batch")
            dice_scores.extend(metric_val)
            all_losses.append(loss)
            _step +=1 
        all_l = torch.mean(torch.stack(all_losses))
        all_d = torch.mean(torch.vstack(dice_scores))
        return all_d, all_l

    def fit(self):
        num_epochs = self.total_epochs 
        best_val_loss = 900000
        for i in range(num_epochs): 
            self.train_epoch()
            self.sch.step()
            self.tb.add_scalar('learning_rate',scalar_value=self.sch.get_lr()[0],global_step=self.c_epoch)
            val_loss = self.val_epoch()
            if val_loss <= best_val_loss: 
                self.store_model()
    def store_model(self):
        model_dir = self.conf['log_dir']
        w_path = os.path.join(model_dir,'model_w.ckpt')
        torch.save({
            'conf':self.conf,
            'model_weights':self.model.state_dict(),
            'epoch':self.c_epoch,
            },f=w_path
        )

@TrainerRegister.register("OneBranchConf")
class OneBranchTrainer(DiceTrainer):
    def __init__(self, model, device='cuda:0', tb_writter=None, conf=None, data_loaders=None):
        super().__init__(model, device, tb_writter, conf, data_loaders) 
    def _build_criterions(self): 
        self.criterions = dict() 
        self.criterions['task'] = DiceCELoss(include_background=True,reduction="mean",to_onehot_y=True,sigmoid=True)
        self.criterions['domain'] = torch.nn.CrossEntropyLoss(reduction="none")
        self.criterions['conf'] = confusion_loss(reduction='none')
        self.criterions['dice'] = DiceCELoss() 
        self.metrics = dict() 
        self.metrics['dice'] = DiceMetric(include_background=False,reduction="mean_batch")
        self.phase_k = 'phase'
    def  _build_optims(self): 
        self.optis = dict()  #TODO: do i just have 1 parameter or do i have a tone of parameters for this 
        lr = self.conf['learn_rate'] 
        mommentum = self.conf['momentum'] 
        self.optis['task'] = optim.Adam(self.model.parameters(),lr=lr,momentum=mommentum)
        print("filtering weights for bottleneck")
        self.optis['domain']= optim.Adam(self.filter_params('bottleneck_branch'),lr=0.1)  
        self.optis['confusion'] = optim.Adam(self.model.parameters(),lr=lr,momentum=mommentum)

    def filter_params(self,param_pattern): 
        param_list = list() 
        for n,e in self.model.named_parameters():
            if param_pattern in n: 
                param_list.append(e)
        return param_list
    def train_epoch(self): 
        self.model.train()
        epoch_loss = 0  
        if self.c_epoch <200: 
            confusion_lambda = 0 
        else: 
            confusion_lambda = 0.1
        for batch_n,batch_data in enumerate(self.tr_dl): 
            inputs, labels,phase = (batch_data[self.img_k], batch_data[self.lbl_k],batch_data[self.phase_k])
            for e in self.optis: 
                self.optis[e].zero_grad() 
            step +=1 
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            has_kidney = (labels.sum(1).sum(1).sum(1).sum(1) !=0).to(torch.int)
            phase = phase.to(self.device) 
            #train the segmentation. entire unet structure
            seg_pred,embed_domain_pred= self.model(inputs) 
            seg_loss = self.criterions['task'](seg_pred,labels)
            embed_domain_loss = (self.criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
            total_loss =  seg_loss + 0.0*embed_domain_loss #we need to do this so dataParallel doesn't yell at me
            total_loss.backward()
            clip_grad_norm(self.model.parameters(),max_norm=2.0,norm_type=2.0)
            self.optis['task'].step()   
            for e in self.optis: 
                self.optis[e].zero_grad() 
            #train the discriminator 
            seg_pred,embed_domain_pred= self.model(inputs) 
            seg_loss = self.criterions['task'](seg_pred,labels)
            embed_domain_loss = (self.criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
            clip_grad_norm(self.model.parameters(),max_norm=2.0,norm_type=2.0)
            total_loss = 0.0*seg_loss + embed_domain_loss #block out the seg loss for ddp calcs
            total_loss.backward()
            self.optis['domain'].step() 
            for e in self.optis: 
                self.optis[e].zero_grad() 
            #penalize the discriminators only  with confusion loss
            seg_pred,embed_domain_pred = self.model(inputs) 
            embed_conf_loss = torch.mean(self.criterions['conf'](embed_domain_pred,phase)*has_kidney)
            seg_loss = self.criterions['task'](seg_pred,labels)
            conf_loss = ( embed_conf_loss ) 
            new_conf_loss = confusion_lambda*(conf_loss + seg_loss)
            new_conf_loss.backward()   
            clip_grad_norm(self.model.parameters(),max_norm=2.0,norm_type=2.0)
            self.optis['confusion'].step()
            for e in self.optis: 
                self.optis[e].zero_grad() 
            #aggrgate all the losses for metrics
            total_loss =  seg_loss.detach() + embed_domain_loss.detach() + conf_loss.detach()
            if torch.isnan(total_loss).any():
                pdb.set_trace()
            epoch_loss += total_loss
            self.tb.add_scalar("batch_DICE+CE_loss",seg_loss,global_step=global_step_count)
            self.tb.add_scalar("batch_total_loss",total_loss,global_step=global_step_count)
            self.tb.add_scalar("embed_domain_loss",embed_domain_loss,global_step=global_step_count)
            self.tb.add_scalar("embed_confusion_loss",embed_conf_loss,global_step=global_step_count)
        global_step_count += 1
        epoch_loss /= step 
        epoch_loss =  epoch_loss.to(self.device)
        return epoch_loss,global_step_count
    
@TrainerRegister.register("OneBranchConfBad")
class OneBranchTrainerBad(OneBranchTrainer):
    def __init__(self, model, device='cuda:0', tb_writter=None, conf=None, data_loaders=None):
        super().__init__(model, device, tb_writter, conf, data_loaders)
    def  _build_optims(self): 
        self.optis = dict()  #TODO: do i just have 1 parameter or do i have a tone of parameters for this 
        lr = self.conf['learn_rate'] 
        mommentum = self.conf['momentum'] 
        self.optis['task'] = optim.Adam(self.model.parameters(),lr=lr,momentum=mommentum)
    def train_epoch(self): 
        self.model.train()
        epoch_loss = 0  
        if self.c_epoch <200: 
            confusion_lambda = 0 
        else: 
            confusion_lambda = 0.1
        for batch_n,batch_data in enumerate(self.tr_dl): 
            inputs, labels,phase = (batch_data[self.img_k], batch_data[self.lbl_k],batch_data[self.phase_k])
            for e in self.optis: 
                self.optis[e].zero_grad() 
            step +=1 
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            has_kidney = (labels.sum(1).sum(1).sum(1).sum(1) !=0).to(torch.int)
            phase = phase.to(self.device) 
            #train the segmentation. entire unet structure
            seg_pred,embed_domain_pred= self.model(inputs) 
            seg_loss = self.criterions['task'](seg_pred,labels)
            embed_domain_loss = (self.criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
            total_loss =  seg_loss + 0.0*embed_domain_loss #we need to do this so dataParallel doesn't yell at me
            total_loss.backward()
            #clip_grad_norm(self.model.parameters(),max_norm=2.0,norm_type=2.0)
            self.optis['task'].step()   
            for e in self.optis: 
                self.optis[e].zero_grad() 
            #train the discriminator 
            seg_pred,embed_domain_pred= self.model(inputs) 
            seg_loss = self.criterions['task'](seg_pred,labels)
            embed_domain_loss = (self.criterions['domain'](embed_domain_pred,phase)*has_kidney).mean()
            clip_grad_norm(self.model.parameters(),max_norm=2.0,norm_type=2.0)
            total_loss = 0.0*seg_loss + embed_domain_loss #block out the seg loss for ddp calcs
            total_loss.backward()
            self.optis['task'].step() 
            for e in self.optis: 
                self.optis[e].zero_grad() 
            #penalize the discriminators only  with confusion loss
            seg_pred,embed_domain_pred = self.model(inputs) 
            embed_conf_loss = torch.mean(self.criterions['conf'](embed_domain_pred,phase)*has_kidney)
            seg_loss = self.criterions['task'](seg_pred,labels)
            conf_loss = ( embed_conf_loss ) 
            new_conf_loss = confusion_lambda*(conf_loss + seg_loss)
            new_conf_loss.backward()   
            #clip_grad_norm(self.model.parameters(),max_norm=2.0,norm_type=2.0)
            self.optis['task'].step()
            for e in self.optis: 
                self.optis[e].zero_grad() 
            #aggrgate all the losses for metrics
            total_loss =  seg_loss.detach() + embed_domain_loss.detach() + conf_loss.detach()
            if torch.isnan(total_loss).any():
                pdb.set_trace()
            epoch_loss += total_loss
            self.tb.add_scalar("batch_DICE+CE_loss",seg_loss,global_step=global_step_count)
            self.tb.add_scalar("batch_total_loss",total_loss,global_step=global_step_count)
            self.tb.add_scalar("embed_domain_loss",embed_domain_loss,global_step=global_step_count)
            self.tb.add_scalar("embed_confusion_loss",embed_conf_loss,global_step=global_step_count)
        global_step_count += 1
        epoch_loss /= step 
        epoch_loss =  epoch_loss.to(self.device)
        return epoch_loss,global_step_count