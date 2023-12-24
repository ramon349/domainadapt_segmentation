import pickle as pkl
from monai.data import Dataset, PersistentDataset


def kit_factory(mode=None):
    fact = None
    if mode == "basic":
        return Dataset
    if mode == "cached":
        return PersistentDataset
