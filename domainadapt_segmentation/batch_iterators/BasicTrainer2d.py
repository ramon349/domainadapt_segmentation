from .BasicTrainer import DiceTrainer 
from monai.metrics import DiceMetric
import copy 
from monai.transforms import AsDiscrete, Compose, Activations
from tqdm import tqdm 
import torch 
from monai.inferers import sliding_window_inference
from torch.nn import PairwiseDistance
from monai.data import decollate_batch




class Dice2DTrainer(DiceTrainer): 
    def __init__(self, model, tb_writter=None, conf=None, dl_dict=None):
        super(Dice2DTrainer,self).__init__(model=model,tb_writter=tb_writter,conf=conf,dl_dict=dl_dict)
        self.reference_model =  copy.deepcopy(self.model)
        self.reference_model.train()
        self.num_domains = 2

    def _get_batch_domain(self,sample_vol): 
        roi_size = self.conf["spacing_vox_dim"]
        dis = 99999999
        best_out = None
        selected_domain = -1
        for  domain_id in range(self.num_domains): 
            pred_func = lambda x: self.reference_model.predictor_wrapper(x,domain_id=domain_id)
            output = sliding_window_inference(
                inputs=sample_vol,  
                roi_size=roi_size,       
                sw_batch_size= 1,          #batch size
                predictor=pred_func, 
                sw_device=self.device, 
                device="cpu" 
            )
            #calculating the updated statistics
            means, vars = get_bn_stats(self.reference_model, domain_id) 
            #i feel like i need toreload my model
            #calculating the distance
            new_dis = cal_distance(means, means_list[domain_id], vars, vars_list[domain_id])
            
            #selcting the best domain
            if new_dis < dis:
                selected_domain = domain_id
                dis = new_dis
        return   selected_domain
        




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
                    val_data[img_k],
                    val_data[lbl_k],
                ) 
                domain_id = self._get_batch_domain(val_inputs)
                pred_func = lambda x: self.model.predictor_wrapper(x,domain_id=domain_id)

                val_data["pred"] = sliding_window_inference(
                    inputs=val_inputs,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=pred_func,
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


def get_bn_stats(model, domain_id):
    means = []
    vars = []
    for name, param in model.state_dict().items():
        if 'bns.{}.running_mean'.format(domain_id) in name:
            means.append(param.clone())
        elif 'bns.{}.running_var'.format(domain_id) in name:
            vars.append(param.clone())
    return means, vars
def _extract_bn_stats(model): 
    means_list = []
    vars_list = []

    for i in range(2):
        means, vars = get_bn_stats(model, i)
        means_list.append(means)
        vars_list.append(vars)
    return means_list,vars_list 
def cal_distance(means_1, means_2, vars_1, vars_2):
    pdist = PairwiseDistance(p=2)
    dis = 0
    for (mean_1, mean_2, var_1, var_2) in zip(means_1, means_2, vars_1, vars_2):
        dis += (pdist(mean_1.reshape(1, mean_1.shape[0]), mean_2.reshape(1, mean_2.shape[0])) + pdist(var_1.reshape(1, var_1.shape[0]), var_2.reshape(1, var_2.shape[0])))
    return dis.item()