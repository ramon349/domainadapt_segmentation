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
from ..helper_utils.losses import ProtoLoss, PseudoLabel_Loss
from einops import rearrange
import pdb
from collections import OrderedDict
import copy
from torch.nn import functional as F
import numpy as np
import pandas as pd
from ..helper_utils.utils import reduce_tensors


@TrainerRegister.register("DiceTrainer")
class DiceTrainer(object):
    def __init__(self, model, tb_writter=None, conf=None, dl_dict=None):
        self.model = model
        self.tb: SummaryWriter = tb_writter
        self.dl_dict = dl_dict
        self.conf = conf
        self.total_epochs = conf["epochs"]
        self.c_epoch = 0
        self.rank = self.conf["rank"] if "rank" in self.conf else 0
        self.world_size = len(self.conf["device"])
        self.device = self.conf["device"][self.rank]
        self.gb_step = 0
        self.img_k = conf["img_key_name"]
        self.lbl_k = conf["lbl_key_name"]
        self.init_optims()
        self._build_criterions()

    def _build_criterions(self):
        self.criterions = dict()
        self.criterions["dice"] = DiceCELoss(to_onehot_y=True)
        self.metrics = dict()
        self.metrics["dice"] = DiceMetric(
            include_background=False, reduction="mean_batch"
        )

    def _log_model_graph(self):
        if self.rank == 0:
            self.model = self.model.eval()
            vox_sample = torch.rand([1, 1] + self.conf["spacing_vox_dim"])
            self.tb.add_graph(self.model, vox_sample.to(self.device))
            self.mdoel = self.model.train()

    def init_optims(self):
        self.opti: optim.SGD = optim.SGD(
            self.model.parameters(), lr=self.conf["learn_rate"]
        )
        self.sch = optim.lr_scheduler.PolynomialLR(
            self.opti, total_iters=self.total_epochs, power=1.5
        )

    def _log_var(self, val_name, val, gb_step):
        avg_val = reduce_tensors(val, world_size=self.world_size).cpu().detach().item()
        if self.rank == 0:
            self.tb.add_scalar(val_name, avg_val, gb_step)

    def train_epoch(self):
        self.model = self.model.train()
        # this is a workaround to have multiple processes updating progress bar
        # only have process 0 update.
        train_batch = enumerate(self.dl_dict["train"])
        if self.rank == 0:
            train_batch = tqdm(train_batch, total=len(self.dl_dict["train"]))
        for i, batch in train_batch:
            inputs, labels = (batch[self.img_k], batch[self.lbl_k])
            self.opti.zero_grad()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterions["dice"](outputs, labels)
            loss.backward()
            self.opti.step()
            self._log_var("t_batch_f_loss", loss, self.gb_step)
            self.opti.zero_grad()
            self.gb_step += 1
        self.c_epoch += 1

    def val_epoch(self):
        self.model = self.model.eval()
        roi_size = self.conf["spacing_vox_dim"]
        num_seg_labels = self.conf["num_seg_labels"]
        metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.model = self.model.eval()
        all_losses = list()
        dice_scores = list()
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        post_label = Compose([Activations(to_onehot=num_seg_labels)])
        _step = 0
        batch_size = self.conf["batch_size"]
        with torch.no_grad():
            for i, val_data in enumerate(self.dl_dict["val"]):
                val_inputs, val_labels = (
                    val_data[self.img_k].to(self.device),
                    val_data[self.lbl_k],
                )
                # this distinciton is needed because my 2D models need a way to compress the 2d patches to be (h,w) instead of (h,w,1).TODO: can i clean htat up?
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=batch_size,
                    predictor=self.model,
                    sw_device=self.device,
                    mode="constant",
                    device="cpu",
                ).to(self.device)
                loss = self.criterions["dice"](val_outputs.to("cpu"), val_labels)
            val_outputs = [
                post_pred(i).to(self.device) for i in decollate_batch(val_outputs)
            ]
            val_labels = [
                post_label(i).to(self.device) for i in decollate_batch(val_labels)
            ]
            if _step == 0 and self.rank == 0:
                write_pred_batches(
                    writer=self.tb,
                    inputs=val_data[self.img_k].detach(),
                    labels=val_labels,
                    preds=val_outputs,
                    epoch=self.c_epoch,
                    dset="val",
                    config=self.conf,
                    is_eval=True,
                )
            metric(y_pred=val_outputs, y=val_labels)
            metric_val = metric.aggregate(reduction="mean_batch").item()
            metric.reset()
            dice_scores.extend(metric_val)
            all_losses.append(loss)
            _step += 1
        all_l = torch.mean(torch.stack(all_losses))
        all_d = torch.mean(torch.vstack(dice_scores))
        return all_d, all_l

    def fit(self):
        num_epochs = self.total_epochs
        best_val_loss = 900000
        for i in range(num_epochs):
            self.train_epoch()
            self.sch.step()
            if self.rank == 0:
                self.tb.add_scalar(
                    "learning_rate",
                    scalar_value=self.sch.get_lr()[0],
                    global_step=self.c_epoch,
                )
            val_dice, val_loss = self.val_epoch()
            if val_loss <= best_val_loss:
                self.store_model()

    def store_model(self):
        model_dir = self.conf["log_dir"]
        w_path = os.path.join(model_dir, "model_w.ckpt")
        torch.save(
            {
                "conf": self.conf,
                "model_weights": self.model.state_dict(),
                "epoch": self.c_epoch,
            },
            f=w_path,
        )

    def test_loop(self, loader, post_t):
        roi_size = self.conf["spacing_vox_dim"]
        img_k = self.conf["img_key_name"]
        lbl_k = self.conf["lbl_key_name"]
        num_seg_labels = self.conf["num_seg_labels"]
        metric = DiceMetric(include_background=True, reduction="mean")
        self.model.eval()
        all_losses = list()
        dice_scores = list()
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        post_label = Compose([Activations(to_onehot=2)])
        _step = 0
        pids = list()
        img_path = list()
        lbl_path = list()
        saved_path = list()
        with torch.no_grad():
            for val_data in tqdm(loader, total=len(loader)):
                val_inputs, val_labels = (
                    val_data[img_k].to(self.device),
                    val_data[lbl_k].to(self.device),
                )
                val_data["pred"] = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=self.model,
                    sw_device=self.device,
                    device="cpu",
                )
                val_data["pred_meta_dict"] = val_data["image_meta_dict"]
                val_outputs = [
                    post_pred(i).to("cpu") for i in decollate_batch(val_data["pred"])
                ]
                val_labels = [
                    post_label(i).to("cpu") for i in decollate_batch(val_labels)
                ]
                val_store = [post_t(i) for i in decollate_batch(val_data)]
                metric(y_pred=val_outputs, y=val_labels)
                metric_val = metric.aggregate().item()
                metric.reset()
                dice_scores.append(metric_val)
                pids.append(val_data["pid"][0])
                stored_path = val_store[0]["pred"].meta["saved_to"]
                saved_path.append(stored_path)
                img_path.append(val_data["image_meta_dict"]["filename_or_obj"][0])
                lbl_path.append(val_data["label_meta_dict"]["filename_or_obj"][0])
            out_df = pd.DataFrame(
                {
                    "pids": pids,
                    "img": img_path,
                    "lbl": lbl_path,
                    "dice": dice_scores,
                    "pred": saved_path,
                }
            )
        return out_df
    def infer_loop(self,post_t): 
        roi_size = self.conf["spacing_vox_dim"]
        img_k = self.conf["img_key_name"]
        self.model = self.model.eval()
        img_path = list()
        saved_path = list()
        with torch.no_grad():
            for val_data in tqdm(self.dl_dict['infer'] , total=len(self.dl_dict['infer'])):
                val_inputs = val_data[img_k].to(self.device)
                val_data["pred"] = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=self.model,
                    sw_device=self.device,
                    device='cpu' 

                )
                val_store = [post_t(i) for i in decollate_batch(val_data)]
                stored_path = [e['pred'].meta['saved_to'] for e in val_store] 
                saved_path.append(stored_path[0])
                img_path.append(val_data["image_meta_dict"]["filename_or_obj"][0])
        out_df = pd.DataFrame({'img':img_path,'pred':saved_path})
        return out_df 


@TrainerRegister.register(cls_name="MiccaiPFA")
class DomainPFATrainerRamen(DiceTrainer):
    def __init__(
        self, model, device="cuda:0", tb_writter=None, conf=None, data_loaders=None
    ):
        super().__init__(model, device, tb_writter, conf, data_loaders)
        self._load_source_model_weights()

    def _load_source_model_weights(self):
        w_path = self.conf["source_model_weight"]
        weights = torch.load(w_path, map_location=self.device)
        new_weights = OrderedDict()
        state_d = weights["state_dict"]
        for n in state_d:
            new_name = n.replace("module.", "")
            new_weights[new_name] = state_d[n]
        self.model.load_state_dict(new_weights)
        self.prototypes = torch.clone(self.model.conv_final[2].conv.weight).view((2, 8))
        for e in self.model.conv_final[2].parameters():
            e.requires_grad = False

    def _build_criterions(self):
        self.criterions = dict()
        pce_config = {
            "use_prototype": True,
            "update_prototype": False,
            "ce_ignore_index": -1,
        }
        self.criterions["dice"] = DiceCELoss(to_onehot_y=True)
        self.criterions["proto_loss"] = ProtoLoss(
            nav_t=0.1, beta=0.001, num_classes=2, device=self.conf["device"][0]
        )
        self.metrics = dict()
        self.metrics["dice"] = DiceMetric(
            include_background=False, reduction="mean_batch"
        )

    def train_epoch(self):
        self.model = self.model.train()
        for i, batch in tqdm(enumerate(self.tr_dl), total=len(self.tr_dl)):
            inputs, labels = (batch[self.img_k], batch[self.lbl_k])
            self.opti.zero_grad()
            inputs = inputs.to(self.device)
            target_f, predict = self.model(inputs, only_feature=True)
            target_f = rearrange(target_f, "b c h w d -> (b h w d) c")
            t2p_loss, p2t_loss = self.criterions["proto_loss"](
                self.prototypes, target_f
            )
            loss = t2p_loss + p2t_loss
            loss.backward()
            self.opti.step()
            self.tb.add_scalar(
                "t_batch_f_loss",
                loss.cpu().detach().item(),
                global_step=self.gb_step,
            )
            self.opti.zero_grad()
            self.gb_step += 1
            if i >= 200:
                break
        self.c_epoch += 1


@TrainerRegister.register(cls_name="MiccaiCL")
class DomainPFATrainerRamen(DiceTrainer):
    def __init__(
        self, model, device="cuda:0", tb_writter=None, conf=None, data_loaders=None
    ):
        super().__init__(model, device, tb_writter, conf, data_loaders)
        self._load_source_model_weights()
        self.total_steps = len(self.tr_dl) * self.total_epochs
        self.model = self.model.to("cpu")
        self.teacher_model = copy.deepcopy(self.model)
        # send models to gpus
        self.teacher_model = self.teacher_model.to("cuda:2")
        self.model = self.model.to(self.device)
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        # bulding the class-wise memmory bank
        self.memobank = []
        self.queue_ptrlis = []
        self.queue_size = []
        num_classes = 2
        output_dim = 8
        for i in range(num_classes):
            self.memobank.append([torch.zeros(0, 8)])
            self.queue_size.append(30000)
            self.queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
        self.queue_size[0] = 50000
        self.momentum_prototype = False
        self.criterion_pseudo = PseudoLabel_Loss()
        self.opt = {
            "use_pseudo": False,
            "drop_percent": 80,
            "use_contra": True,
            "use_source_prototypes": False,
            "momentum_prototype": False,
            "num_queries": 64,
            "num_negatives": 256,
            "current_class_threshold": 0.3,
            "low_entropy_threshold": 20,
            "low_rank": 2,
            "high_rank": 5,
            "num_classes": 2,
            "gpu_id": self.device,
        }

    def _load_source_model_weights(self):
        w_path = self.conf["source_model_weight"]
        weights = torch.load(w_path, map_location=self.device)
        new_weights = OrderedDict()
        state_d = weights["model_weights"]
        for n in state_d:
            new_name = n.replace("module.", "")
            new_weights[new_name] = state_d[n]
        self.model.load_state_dict(new_weights)
        self.prototypes = torch.clone(self.model.conv_final[2].conv.weight).view((2, 8))
        self.source_prototypes = torch.clone(self.model.conv_final[2].conv.weight).view(
            (2, 8)
        )
        for e in self.model.conv_final[2].parameters():
            e.requires_grad = False

    def train_epoch(self):
        self.model = self.model.train()
        for i, batch in tqdm(enumerate(self.tr_dl), total=len(self.tr_dl)):
            inputs, labels = (batch[self.img_k], batch[self.lbl_k])
            self.opti.zero_grad()
            inputs = inputs.to(self.device)
            labels = inputs.to(self.device)
            predict_teacher, predicts, losses = self.train_one_step(inputs, labels)

    def train_one_step(self, imgs, labels):
        low_entropy_threshold = self.opt["low_entropy_threshold"]
        drop_percent = self.opt["drop_percent"]
        self.opti.zero_grad()
        alpha_t = low_entropy_threshold * (1 - self.gb_step / self.total_steps)
        target_f_teacher, pred_teacher = self.teacher_model(imgs, only_feature=True)
        prob_teacher = F.softmax(
            pred_teacher, dim=1
        )  # TODO: you  might want to hcange this with softmax
        pseudo_label_teacher = torch.argmax(pred_teacher, dim=1)
        # loss calculation
        drop_percent = self.opt["drop_percent"]
        percent_unreliable = (100 - drop_percent) * (
            1 - self.gb_step / self.total_steps
        )
        drop_percent = 100 - percent_unreliable

        target_f, pred = self.model(imgs, only_feature=True)
        total_loss = 0
        adapt_losses = {}
        if self.opt["use_pseudo"]:
            pesudo_unsup_loss = self.criterion_pseudo(
                pred, pseudo_label_teacher.clone(), drop_percent, prob_teacher.detach()
            )
            total_loss += pesudo_unsup_loss
            adapt_losses["pseudo_loss"] = pesudo_unsup_loss.detach()
        if self.opt["use_contra"]:
            with torch.no_grad():
                entropy = -torch.sum(
                    prob_teacher * torch.log(prob_teacher + 1e-10), dim=1
                )
                low_entropy_mask = 0
                high_entropy_mask = 0
                for i in range(pred.shape[1]):
                    if torch.sum(entropy[pseudo_label_teacher == i]) > 5:
                        if i == 0:
                            low_thresh, high_thresh = np.percentile(
                                entropy[pseudo_label_teacher == i]
                                .detach()
                                .cpu()
                                .numpy()
                                .flatten(),
                                [90 - alpha_t / 4, 100 - alpha_t / 4],
                            )
                        else:
                            low_thresh, high_thresh = np.percentile(
                                entropy[pseudo_label_teacher == i]
                                .detach()
                                .cpu()
                                .numpy()
                                .flatten(),
                                [90 - alpha_t, 100 - alpha_t],
                            )
                        high_entropy_mask = (
                            high_entropy_mask
                            + entropy.ge(high_thresh).bool()
                            * (pseudo_label_teacher == i).bool()
                        )
                        low_entropy_mask = (
                            low_entropy_mask
                            + entropy.le(low_thresh).bool()
                            * (pseudo_label_teacher == i).bool()
                        )

            new_keys, contra_loss, valid_classes, pos_masks, neg_masks = (
                self.compute_contra_memobank_loss(
                    target_f,
                    self.label_onehot(pseudo_label_teacher),
                    prob_teacher,
                    low_entropy_mask.unsqueeze(1),
                    high_entropy_mask.unsqueeze(1),
                    target_f_teacher,
                )
            )
            total_loss += contra_loss
            adapt_losses["contra_loss"] = contra_loss.detach()

        total_loss.backward()
        self.opti.step()

        adapt_losses = {}

        adapt_losses["total_loss"] = total_loss.detach()

        return pred_teacher, pred, adapt_losses

    def compute_contra_memobank_loss(
        self,
        target_f,
        pseudo_label_teacher,
        prob_teacher,
        low_mask,
        high_mask,
        target_f_teacher,
    ):
        # current_class_threshold: delta_p (0.3)
        current_class_threshold = self.opt["current_class_threshold"]
        low_rank, high_rank = self.opt["low_rank"], self.opt["high_rank"]
        temp = 0.1
        num_queries = self.opt["num_queries"]
        num_negatives = self.opt["num_negatives"]

        num_feat = target_f.shape[1]
        low_valid_pixel = pseudo_label_teacher * low_mask
        # high_valid_pixel = pseudo_label_teacher * high_mask

        target_f = target_f.permute(0, 2, 3, 4, 1)
        target_f_teacher = target_f_teacher.permute(0, 2, 3, 4, 1)

        seg_feat_all_list = []
        seg_feat_low_entropy_list = []  # candidate anchor pixels
        seg_num_list = []  # the number of low_valid pixels in each class
        seg_proto_list = []  # the center of each class

        _, prob_indices_teacher = torch.sort(prob_teacher, 1, True)
        prob_indices_teacher = prob_indices_teacher.permute(
            0, 2, 3, 4, 1
        )  # (num_unlabeled, h, w, num_cls)

        valid_classes = []
        positive_masks = []
        negative_masks = []
        new_keys = []

        for i in range(self.opt["num_classes"]):
            low_valid_pixel_seg = low_valid_pixel[
                :, i
            ]  # select binary mask for i-th class
            prob_seg = prob_teacher[:, i, :, :]
            target_f_mask_low_entropy = (
                prob_seg > current_class_threshold
            ) * low_valid_pixel_seg.bool()

            seg_feat_all_list.append(target_f[low_valid_pixel_seg.bool()])
            seg_feat_low_entropy_list.append(target_f[target_f_mask_low_entropy])

            # positive sample: center of the class
            seg_proto_list.append(
                torch.mean(
                    target_f_teacher[low_valid_pixel_seg.bool()].detach(),
                    dim=0,
                    keepdim=True,
                )
            )

            # generate class mask for unlabeled data
            # prob_i_classes = prob_indices_teacher[target_f_mask_high_entropy[num_labeled :]]
            class_mask = torch.sum(
                prob_indices_teacher[:, :, :, :, low_rank:high_rank].eq(i), dim=4
            ).bool()

            negative_mask = high_mask[:, 0].bool() * class_mask

            keys = target_f_teacher[negative_mask].detach()

            new_keys.append(self.dequeue_and_enqueue(keys=keys, class_idx=i))

            if low_valid_pixel_seg.sum() > 0:
                seg_num_list.append(int(low_valid_pixel_seg.sum().item()))
                valid_classes.append(i)
                positive_masks.append(target_f_mask_low_entropy)
                negative_masks.append(negative_mask)

        if (
            len(seg_num_list) <= 1
        ):  # in some rare cases, a small mini-batch might only contain 1 or no semantic class

            return (
                new_keys,
                torch.tensor(0.0) * target_f.sum(),
                valid_classes,
                positive_masks,
                negative_masks,
            )

        else:
            reco_loss = torch.tensor(0.0).to(self.device)
            seg_proto = torch.cat(seg_proto_list)  # shape: [valid_seg, 256]
            valid_seg = len(seg_num_list)  # number of valid classes

            prototype = torch.zeros(
                (prob_indices_teacher.shape[-1], num_queries, 1, num_feat)
            ).to(self.opt["gpu_id"])

            for i in range(valid_seg):
                if (
                    len(seg_feat_low_entropy_list[i]) > 0
                    and self.memobank[valid_classes[i]][0].shape[0] > 0
                ):
                    # select anchor pixel
                    seg_low_entropy_idx = torch.randint(
                        len(seg_feat_low_entropy_list[i]), size=(num_queries,)
                    )
                    anchor_feat = (
                        seg_feat_low_entropy_list[i][seg_low_entropy_idx]
                        .clone()
                        .to(self.opt["gpu_id"])
                    )
                else:
                    # in some rare cases, all queries in the current query class are easy
                    reco_loss = reco_loss + 0 * target_f.sum()
                    continue

                # apply negative key sampling from memory bank (with no gradients)
                with torch.no_grad():
                    negative_feat = (
                        self.memobank[valid_classes[i]][0]
                        .clone()
                        .to(self.opt["gpu_id"])
                    )

                    high_entropy_idx = torch.randint(
                        len(negative_feat), size=(num_queries * num_negatives,)
                    )
                    negative_feat = negative_feat[high_entropy_idx]
                    negative_feat = negative_feat.reshape(
                        num_queries, num_negatives, num_feat
                    )
                    if self.opt["use_source_prototypes"]:
                        positive_feat = (
                            self.source_prototypes[i]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .repeat(num_queries, 1, 1)
                            .to(self.opt["gpu_id"])
                        )  # (num_queries, 1, num_feat)

                    else:
                        positive_feat = (
                            seg_proto[i]
                            .unsqueeze(0)
                            .unsqueeze(0)
                            .repeat(num_queries, 1, 1)
                            .to(self.opt["gpu_id"])
                        )  # (num_queries, 1, num_feat)
                        if self.momentum_prototype is not None:
                            if not (self.momentum_prototype == 0).all():
                                ema_decay = min(
                                    1
                                    - 1
                                    / (
                                        self.iter_counter.steps_so_far
                                        / self.opt["batch_size"]
                                    ),
                                    0.999,
                                )
                                positive_feat = (
                                    (1 - ema_decay) * positive_feat
                                    + ema_decay
                                    * self.momentum_prototype[valid_classes[i]]
                                )

                            prototype[valid_classes[i]] = positive_feat.clone()

                    all_feat = torch.cat(
                        (positive_feat, negative_feat), dim=1
                    )  # (num_queries, 1 + num_negative, num_feat)

                seg_logits = torch.cosine_similarity(
                    anchor_feat.unsqueeze(1), all_feat, dim=2
                )
                # pdb.set_trace()

                reco_loss = reco_loss + F.cross_entropy(
                    seg_logits / temp,
                    torch.zeros(num_queries).long().to(self.opt["gpu_id"]),
                )

            if self.momentum_prototype is None:
                return (
                    new_keys,
                    reco_loss / valid_seg,
                    valid_classes,
                    positive_masks,
                    negative_masks,
                )

            else:
                self.momentum_prototype = prototype
                return (
                    new_keys,
                    reco_loss / valid_seg,
                    valid_classes,
                    positive_masks,
                    negative_masks,
                )

    def label_onehot(self, inputs):
        one_hot_inputs = F.one_hot(inputs, self.opt["num_classes"])
        return one_hot_inputs.permute(0, 4, 1, 2, 3)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, class_idx):

        queue = self.memobank[class_idx]
        queue_ptr = self.queue_ptrlis[class_idx]
        queue_size = self.queue_size[class_idx]
        keys = keys.clone().cpu()

        batch_size = keys.shape[0]

        ptr = int(queue_ptr)

        queue[0] = torch.cat((queue[0], keys.cpu()), dim=0)
        if queue[0].shape[0] >= queue_size:
            queue[0] = queue[0][-queue_size:, :]
            ptr = queue_size
        else:
            ptr = (ptr + batch_size) % queue_size  # move pointer

        queue_ptr[0] = ptr

        return batch_size

    def val_epoch(self):
        self.model = self.model.eval()
        roi_size = self.conf["spacing_vox_dim"]
        num_seg_labels = self.conf["num_seg_labels"]
        metric = DiceMetric(include_background=False, reduction="mean_batch")
        self.model = self.model.eval()
        all_losses = list()
        dice_scores = list()
        post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        post_label = Compose([Activations(to_onehot=num_seg_labels)])
        _step = 0
        batch_size = self.conf["batch_size"]
        with torch.no_grad():
            for i, val_data in enumerate(self.val_dl):
                val_inputs, val_labels = (
                    val_data[self.img_k].to(self.device),
                    val_data[self.lbl_k],
                )
                # this distinciton is needed because my 2D models need a way to compress the 2d patches to be (h,w) instead of (h,w,1).TODO: can i clean htat up?
                val_outputs = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=batch_size,
                    predictor=self.model,
                    sw_device=self.device,
                    mode="constant",
                    device="cpu",
                ).to(self.device)
                loss = self.criterions["dice"](val_outputs.to("cpu"), val_labels)
            val_outputs = [post_pred(i).to("cpu") for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i).to("cpu") for i in decollate_batch(val_labels)]
            if _step == 0:
                write_pred_batches(
                    writer=self.tb,
                    inputs=val_data[self.img_k].detach(),
                    labels=val_labels,
                    preds=val_outputs,
                    epoch=self.c_epoch,
                    dset="val",
                    config=self.conf,
                    is_eval=True,
                )
            metric(y_pred=val_outputs, y=val_labels)
            metric_val = metric.aggregate(reduction="mean_batch")
            dice_scores.extend(metric_val)
            all_losses.append(loss)
            _step += 1
        all_l = torch.mean(torch.stack(all_losses))
        all_d = torch.mean(torch.vstack(dice_scores))
        return all_d, all_l
