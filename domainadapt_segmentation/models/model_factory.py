import torch 
from ..helper_utils.utils import remove_ddp_tags


class ModelRegister: 
    __data = {}
    @staticmethod
    def __models():
        if not hasattr(ModelRegister,'_data'):
            ModelRegister._data = {} 
        return ModelRegister._data 
    @classmethod
    def register(cls,cls_name=None):
        def decorator(cls_obj):
            cls.__data[cls_name]=cls_obj
            return cls_obj
        return decorator
    @classmethod
    def get_model(cls,key):
        return cls.__data[key]
    @classmethod
    def num_models(cls):
        return len(cls.__data)
    @classmethod
    def get_models(cls):
        return cls.__data.keys()
    @classmethod
    def add_model(cls,key,val):
        cls.__data[key]=val

def model_loader(model_name):
    return ModelRegister.get_model(model_name)()

def get_kernels_strides(patch_size,spacing):
    """
    This function is only used for decathlon datasets with the provided patch sizes.
    When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = patch_size, spacing
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    return kernels, strides

def model_factory(config):
    model_name = config["model"]
    num_seg_labels = config["num_seg_labels"]
    model_func = ModelRegister.get_model(model_name) 
    net = model_func(spatial_dims=3,in_channels=1,out_channels=num_seg_labels)
    if 'model_weight' in config and config['model_weight']: 
        print('loading weights')
        checkpoint= torch.load(config['model_weight'],map_location='cpu') 
        new_d = remove_ddp_tags(checkpoint['state_dict']) 
        net.load_state_dict(new_d,strict=False)
    return net 
