import os
import sys
import torch
import numpy as np
import helper_utils.utils as help_utils
from monai.transforms import AsDiscrete, Compose
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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
                    dset='train'
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


def eval_loop(model, loader, writer, epoch, dset_name, config):
    roi_size = config["spacing_vox_dim"]
    img_k = config["img_key_name"]
    lbl_k = config["lbl_key_name"]
    device = config["device"]
    num_seg_labels = config["num_seg_labels"]
    metric = DiceMetric(include_background=True, reduction="mean")
    model.eval()
    loss_function = DiceLoss(include_background=True, reduction="mean")
    all_losses = list()
    dice_scores = list()
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=num_seg_labels)])
    post_label = Compose([AsDiscrete(to_onehot=num_seg_labels)])
    _step = 0 
    with torch.no_grad():
        for val_data in tqdm(loader, total=len(loader)):
            if _step==0 and epoch==0: 
                help_utils.write_batches(
                    writer=writer,
                    inputs=val_data[img_k].detach(),
                    labels=val_data[lbl_k].detach(),
                    epoch=epoch,
                    dset='val'
                )
            val_inputs, val_labels = (
                val_data[img_k].to(device),
                val_data[lbl_k].to(device),
            )
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=roi_size,
                sw_batch_size=1,
                predictor=model,
            )
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            metric(y_pred=val_outputs, y=val_labels)
            metric_val = metric.aggregate().item()
            metric.reset()
            dice_scores.append(metric_val)
            loss = sum(
                loss_function(v_o, v_l) for v_o, v_l in zip(val_outputs, val_labels)
            )
            all_losses.append(loss)
        all_l = np.array(torch.mean(torch.stack(all_losses)).cpu())
        all_d = np.mean(dice_scores)
        writer.add_scalar(f"test_{dset_name}_dice", all_d, global_step=epoch)
        writer.add_scalar(f"test_{dset_name}_loss", all_l, global_step=epoch)
    return all_d, all_l
