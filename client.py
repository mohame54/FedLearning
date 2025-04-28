#import flwr as fl
#from flwr.common import Context
from utils import (
    get_parameters,
    set_parameters,
    create_opt,
    train_epoch,
    val_epoch,
    load_json,
    write_json
)
from torch.cuda.amp import GradScaler
import torch
import numpy as np
from collections import defaultdict
from models import get_model


class GWNETClient:#(fl.client.NumPyClient):
    def __init__(self, partition_id, adj_met, model_config, train_loader, val_loader):
        super().__init__()
        self.partition_id = partition_id
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(adj_met, **model_config)
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        self.model = self.model.to("cpu")
        return get_parameters(self.model)
    
    def set_parameters(self, parameters):
        print(f"[Client {self.partition_id}] set_parameters")
        self.model = self.model.to("cpu")
        set_parameters(self.model, parameters)

    def fit(self, parameters, config):
       try:
            self.set_parameters(parameters)
            self.model = self.model.to(self.dev)
            if config['optim_type'] == "adamw":
                optim = create_opt(self.model, **config['optimizer_kwargs'])
            else:
                optim = torch.optim.Adam(self.model.parameters(), **config['optimizer_kwargs'])
            
            scaler = None
            if config['mixed_pre']:
                scaler = GradScaler()
            max_norm = config.get("max_norm")
            result_json = config.get("results")
            norm_params = np.load(config['norm_params_pth'])
            norm_params =  {"mean":torch.tensor(norm_params['mean']), "std":torch.tensor(norm_params['std'])}

            for epoch in range(config['epochs']):
                print(f"Training Epoch {epoch +1}/ {config['epochs']}")
                train_mets = train_epoch(
                    self.model,
                    self.train_loader,
                    optim,
                    self.dev,
                    scaler,
                    max_norm=max_norm,
                    grad_accumelation=config['grad_accum'],
                    norms_params=norm_params,
                    mu=config['mu'],
                    is_prox=config['is_prox']
                )
                if result_json is not None:
                    met_json = load_json(result_json['pth'])
                    if met_json is None:
                        met_json = {self.partition_id:{"train":defaultdict(list), "val":defaultdict(list)}}
                    for met in ['loss', "mape", "rmse"]:
                            met_json[self.partition_id]['train'][met].append(train_mets[met])
                    write_json(result_json['pth'], met_json)
                print("-"*50)
            
            self.model = self.model.to("cpu")
            return self.get_parameters(self.model), len(self.train_loader), train_mets
       except Exception as e:
            print(f"[Client {self.partition_id}] ❌ Fit failed: {e}")
            raise
            #return parameters, 0, {"error": str(e)}
           
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model = self.model.to(self.dev)

        norm_params = np.load(config['norm_params_pth'])
        norm_params =  {"mean":torch.tensor(norm_params['mean']), "std":torch.tensor(norm_params['std'])}
        val_mets = val_epoch(
            self.model,
            self.val_loader,
            self.dev,
            norm_params
        )
        result_json = config.get("results")
        if result_json is not None:
               met_json = load_json(result_json['pth'])
               if met_json is None:
                   met_json = {self.partition_id:{"train":defaultdict(list), "val":defaultdict(list)}}
               for met in ['loss', "mape", "rmse"]:
                    met_json[self.partition_id]['val'][met].append(val_mets[met])
               write_json(result_json['pth'], met_json)
        self.model = self.model.to("cpu")
        return float(val_mets['loss']), len(self.val_loader), val_mets
    


def create_client_fn(train_loaders, val_loaders, adj, model_kwargs):
    def client_fn(context):
        cid = context.node_config["partition-id"]
        print(f"🌸 [client_fn] Initializing client {cid}")
        client = GWNETClient(
            partition_id=cid,
            adj_met=adj,
            model_config=model_kwargs,
            train_loader=train_loaders[int(cid)],
            val_loader=val_loaders[int(cid)]
        )
        return client.to_client()
    return client_fn
