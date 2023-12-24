import pickle as pkl


def load_data(data_path):
    # all code will assume this has (train,test,val)
    with open(data_path, "rb") as f:
        all_data = pkl.load(f)
    return all_data
