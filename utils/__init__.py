from .data_utils import prepare_data
from .metrics import (
    masked_mae,
    masked_mape,
    mae_loss,
    mape_error,
    rmse_error
)
from .train_utils import *
import scipy.sparse as sp
import json
import yaml
import copy


def asym_adj(adj):
    # random walk
    adj = sp.coo_matrix(adj)
    # rowsum(A)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def get_adj_mets(data_file):
    data = np.load(data_file)
    adj_mx, mat = data['adj'], data['mat']
    adj = (torch.tensor(asym_adj(adj_mx)),
           torch.tensor(asym_adj(np.transpose(adj_mx)))) # Adaptive,  Distance
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = [t.to(dev) for t in adj]
    return adj, mat


def load_json(json_path):
    try:
        with open(json_path, "rb") as f:
            data = json.load(f)
            return data 
    except FileNotFoundError:
        return None


def write_json(json_path, data, indent=4):
    with open(json_path, "w") as f:
        json.dump(data, f, indent=indent)


class DataCalss:
    def __init__(self, **kwargs):
        self.kwargs = copy.deepcopy(kwargs)
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = DataCalss(**v)
            setattr(self, k, v)

    @classmethod
    def from_yaml_file(cls, yaml_pth):
        data = yaml.safe_load(open(yaml_pth, "r"))
        return cls(**data)
    
    @classmethod
    def from_json_file(cls, json_pth):
        return cls(**load_json(json_pth))
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __setitem__(self, key, val):
        setattr(key, val)

    def as_dict(self):
        return self.kwargs