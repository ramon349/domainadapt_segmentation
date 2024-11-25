
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch import optim
from tqdm import tqdm
import torch
from monai.transforms import AsDiscrete, Compose, Activations
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from ..helper_utils.utils import write_pred_batches
import os
from ..helper_utils.utils import confusion_loss
from torch.nn.utils import clip_grad_norm
import pdb
from .trainer_factory import TrainerRegister
import pdb
from ..helper_utils.utils import reduce_tensors
from .BasicTrainer import DiceTrainer
@TrainerRegister.register("OneBranchConf")
class OneBranchTrainer(DiceTrainer):
    def __init__(
        self, model, tb_writter=None, conf=None, dl_dict=None
    ):
        super().__init__(model, tb_writter=tb_writter, conf=conf, dl_dict=dl_dict)

    def _build_criterions(self):
        self.criterions = dict()
        self.criterions["task"] = DiceCELoss(
            include_background=True, reduction="mean", to_onehot_y=True, sigmoid=True
        )
        self.criterions["domain"] = torch.nn.CrossEntropyLoss(reduction="none")
        self.criterions["conf"] = confusion_loss(reduction="none")
        self.criterions["dice"] = DiceCELoss()
        self.metrics = dict()
        self.metrics["dice"] = DiceMetric(
            include_background=False, reduction="mean_batch"
        )
        self.phase_k = "phase"

    def init_optims(self):
        self.optis = (
            dict()
        )  # TODO: do i just have 1 parameter or do i have a tone of parameters for this
        lr = self.conf["learn_rate"]
        mommentum = self.conf["momentum"]
        self.optis["task"] = optim.SGD(
            self.model.parameters(), lr=lr, momentum=mommentum
        )
        print("filtering weights for bottleneck")
        self.optis["domain"] = optim.SGD(
            self.filter_params("bottleneck_branch"), lr=0.1
        )
        self.optis["confusion"] = optim.SGD(
            self.model.parameters(), lr=lr, momentum=mommentum
        ) 
        self.sch = torch.optim.lr_scheduler.PolynomialLR(self.optis['task'],total_iters=self.total_epochs,power=1.5)


    def filter_params(self, param_pattern):
        param_list = list()
        for n, e in self.model.named_parameters():
            if param_pattern in n:
                param_list.append(e)
        return param_list

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        if self.c_epoch < 200:
            confusion_lambda = 0
        else:
            confusion_lambda = 0.1
        for batch_n, batch_data in enumerate(self.dl_dict['train']):
            inputs, labels, phase = (
                batch_data[self.img_k],
                batch_data[self.lbl_k],
                batch_data[self.phase_k],
            )
            for e in self.optis:
                self.optis[e].zero_grad()
            self.gb_step +=1 
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            has_kidney = (labels.sum(1).sum(1).sum(1).sum(1) != 0).to(torch.int)
            phase = phase.to(self.device)
            # train the segmentation. entire unet structure
            seg_pred, embed_domain_pred = self.model(inputs)
            seg_loss = self.criterions["task"](seg_pred, labels)
            embed_domain_loss = (
                self.criterions["domain"](embed_domain_pred, phase) * has_kidney
            ).mean()
            total_loss = (
                seg_loss + 0.0 * embed_domain_loss
            )  # we need to do this so dataParallel doesn't yell at me
            total_loss.backward()
            clip_grad_norm(self.model.parameters(), max_norm=2.0, norm_type=2.0)
            self.optis["task"].step()
            for e in self.optis:
                self.optis[e].zero_grad()
            # train the discriminator
            seg_pred, embed_domain_pred = self.model(inputs)
            seg_loss = self.criterions["task"](seg_pred, labels)
            embed_domain_loss = (
                self.criterions["domain"](embed_domain_pred, phase) * has_kidney
            ).mean()
            clip_grad_norm(self.model.parameters(), max_norm=2.0, norm_type=2.0)
            total_loss = (
                0.0 * seg_loss + embed_domain_loss
            )  # block out the seg loss for ddp calcs
            total_loss.backward()
            self.optis["domain"].step()
            for e in self.optis:
                self.optis[e].zero_grad()
            # penalize the discriminators only  with confusion loss
            seg_pred, embed_domain_pred = self.model(inputs)
            embed_conf_loss = torch.mean(
                self.criterions["conf"](embed_domain_pred, phase) * has_kidney
            )
            seg_loss = self.criterions["task"](seg_pred, labels)
            conf_loss = embed_conf_loss
            new_conf_loss = confusion_lambda * (conf_loss + seg_loss)
            new_conf_loss.backward()
            clip_grad_norm(self.model.parameters(), max_norm=2.0, norm_type=2.0)
            self.optis["confusion"].step()
            for e in self.optis:
                self.optis[e].zero_grad()
            # aggrgate all the losses for metrics
            total_loss = (
                seg_loss.detach() + embed_domain_loss.detach() + conf_loss.detach()
            )
            if torch.isnan(total_loss).any():
                pdb.set_trace()
            epoch_loss += total_loss
            self._log_var(
                "batch_DICE+CE_loss", seg_loss, global_step=global_step_count
            )
            self._log_var(
                "batch_total_loss", total_loss, global_step=global_step_count
            )
            self._log_var(
                "embed_domain_loss", embed_domain_loss, global_step=global_step_count
            )
            self._log_var(
                "embed_confusion_loss", embed_conf_loss, global_step=global_step_count
            )
        global_step_count += 1
        epoch_loss /= step
        epoch_loss = epoch_loss.to(self.device)
        return epoch_loss, global_step_count


@TrainerRegister.register("OneBranchConfBad")
class OneBranchTrainerBad(OneBranchTrainer):
    def __init__(
        self, model, device="cuda:0", tb_writter=None, conf=None, data_loaders=None
    ):
        super().__init__(model, device, tb_writter, conf, data_loaders)

    def _build_optims(self):
        self.optis = (
            dict()
        )  # TODO: do i just have 1 parameter or do i have a tone of parameters for this
        lr = self.conf["learn_rate"]
        mommentum = self.conf["momentum"]
        self.optis["task"] = optim.Adam(
            self.model.parameters(), lr=lr, momentum=mommentum
        )

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0
        if self.c_epoch < 200:
            confusion_lambda = 0
        else:
            confusion_lambda = 0.1
        for batch_n, batch_data in enumerate(self.tr_dl):
            inputs, labels, phase = (
                batch_data[self.img_k],
                batch_data[self.lbl_k],
                batch_data[self.phase_k],
            )
            for e in self.optis:
                self.optis[e].zero_grad()
            step += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            has_kidney = (labels.sum(1).sum(1).sum(1).sum(1) != 0).to(torch.int)
            phase = phase.to(self.device)
            # train the segmentation. entire unet structure
            seg_pred, embed_domain_pred = self.model(inputs)
            seg_loss = self.criterions["task"](seg_pred, labels)
            embed_domain_loss = (
                self.criterions["domain"](embed_domain_pred, phase) * has_kidney
            ).mean()
            total_loss = (
                seg_loss + 0.0 * embed_domain_loss
            )  # we need to do this so dataParallel doesn't yell at me
            total_loss.backward()
            # clip_grad_norm(self.model.parameters(),max_norm=2.0,norm_type=2.0)
            self.optis["task"].step()
            for e in self.optis:
                self.optis[e].zero_grad()
            # train the discriminator
            seg_pred, embed_domain_pred = self.model(inputs)
            seg_loss = self.criterions["task"](seg_pred, labels)
            embed_domain_loss = (
                self.criterions["domain"](embed_domain_pred, phase) * has_kidney
            ).mean()
            clip_grad_norm(self.model.parameters(), max_norm=2.0, norm_type=2.0)
            total_loss = (
                0.0 * seg_loss + embed_domain_loss
            )  # block out the seg loss for ddp calcs
            total_loss.backward()
            self.optis["task"].step()
            for e in self.optis:
                self.optis[e].zero_grad()
            # penalize the discriminators only  with confusion loss
            seg_pred, embed_domain_pred = self.model(inputs)
            embed_conf_loss = torch.mean(
                self.criterions["conf"](embed_domain_pred, phase) * has_kidney
            )
            seg_loss = self.criterions["task"](seg_pred, labels)
            conf_loss = embed_conf_loss
            new_conf_loss = confusion_lambda * (conf_loss + seg_loss)
            new_conf_loss.backward()
            # clip_grad_norm(self.model.parameters(),max_norm=2.0,norm_type=2.0)
            self.optis["task"].step()
            for e in self.optis:
                self.optis[e].zero_grad()
            # aggrgate all the losses for metrics
            total_loss = (
                seg_loss.detach() + embed_domain_loss.detach() + conf_loss.detach()
            )
            if torch.isnan(total_loss).any():
                pdb.set_trace()
            epoch_loss += total_loss
            self.tb.add_scalar(
                "batch_DICE+CE_loss", seg_loss, global_step=global_step_count
            )
            self.tb.add_scalar(
                "batch_total_loss", total_loss, global_step=global_step_count
            )
            self.tb.add_scalar(
                "embed_domain_loss", embed_domain_loss, global_step=global_step_count
            )
            self.tb.add_scalar(
                "embed_confusion_loss", embed_conf_loss, global_step=global_step_count
            )
        global_step_count += 1
        epoch_loss /= step
        epoch_loss = epoch_loss.to(self.device)
        return epoch_loss, global_step_count