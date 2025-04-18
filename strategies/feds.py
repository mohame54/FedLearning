import torch
import torch
import numpy as np
from collections import defaultdict
from .base import BaseFedStrategy


class FedAvg(BaseFedStrategy):
    def __init__(self, clients, config):
        super().__init__(clients, config)

    def aggregate_parameters(self):
        client_parameters = []
        client_sizes = []
        for client in self.clients:
            client_parameters.append(client.get_parameters(self.config))
            client_sizes.append(len(client.train_loader))

        total_size = sum(client_sizes)
        aggregated_params = {}

        for param_name in client_parameters[0].keys():
            aggregated_params[param_name] = torch.zeros_like(client_parameters[0][param_name])
            for client_params, size in zip(client_parameters, client_sizes):
                param = client_params[param_name]
                aggregated_params[param_name] += (param * torch.tensor(size / total_size, device=param.device, dtype=param.dtype))
        
        return aggregated_params

    def fit(self):
        global_params = self.aggregate_parameters()

        # Set global parameters to each client
        for client in self.clients:
            client.set_parameters(global_params)

        # Train each client
        fit_results = []
        for client in self.clients:
            fit_results.append(client.fit(global_params, self.config))

        # Aggregate the results
        aggregated_metrics = defaultdict(list)
        for result in fit_results:
            train_metrics = result[2]
            for met in train_metrics:
                aggregated_metrics[met].append(np.mean(train_metrics[met]))

        return aggregated_metrics

    def evaluate(self):
        global_params = self.aggregate_parameters()


        # Set global parameters to each client
        for client in self.clients:
            client.set_parameters(global_params)

        # Evaluate each client
        eval_results = []
        for client in self.clients:
            eval_results.append(client.evaluate(global_params, self.config))

        # Aggregate the results (validation metrics)
        aggregated_metrics = defaultdict(list)
        for result in eval_results:
            val_metrics = result[2]
            for met in val_metrics:
                aggregated_metrics[met].append(np.mean(val_metrics[met]))

        return aggregated_metrics


class FedProx(BaseFedStrategy):
    def __init__(self, clients, config):
        super().__init__(clients, config)
    
    def aggregate_parameters(self):
        client_parameters = []
        client_sizes = []
        
        for client in self.clients:
            client_parameters.append(client.get_parameters(self.config))
            client_sizes.append(len(client.train_loader))
        
        total_size = sum(client_sizes)
        aggregated_params = {}
        
        for param_name in client_parameters[0].keys():
            aggregated_params[param_name] = torch.zeros_like(client_parameters[0][param_name])
            for client_params, size in zip(client_parameters, client_sizes):
                param = aggregated_params[param_name]
                aggregated_params[param_name] = param +  (client_params[param_name] * torch.tensor(size / total_size, device=param.device, dtype=param.dtype))
        
        return aggregated_params
    
    def fit(self):
        global_params = self.aggregate_parameters()
        
        # Set global parameters to each client
        for client in self.clients:
            client.set_parameters(global_params)
        
        # Train each client with proximal term
        fit_results = []

        fit_results = []
        for client in self.clients:
            fit_results.append(client.fit(global_params, self.config))

        # Aggregate the results
        aggregated_metrics = defaultdict(list)
        for result in fit_results:
            train_metrics = result[2]
            for met in train_metrics:
                aggregated_metrics[met].append(np.mean(train_metrics[met]))
        
        return aggregated_metrics
    
    def evaluate(self):
        global_params = self.aggregate_parameters()
        
        # Set global parameters to each client
        for client in self.clients:
            client.set_parameters(global_params)
        
        # Evaluate each client
        eval_results = []
        for client in self.clients:
            eval_results.append(client.evaluate(global_params, self.config))
        
        # Aggregate the results (validation metrics)
        aggregated_metrics = defaultdict(list)
        for result in eval_results:
            val_metrics = result[2]
            for met in val_metrics:
                aggregated_metrics[met].append(np.mean(val_metrics[met]))
        
        return aggregated_metrics


class FedNova(BaseFedStrategy):
    def __init__(self, clients, config):
        super().__init__(clients, config)
    
    def aggregate_parameters(self):
        client_updates = []
        client_sizes = []
        client_steps = []

        global_params = self.global_model.state_dict()

        # Collect updates and metadata from clients
        for client in self.clients:
            client_params = client.get_parameters(self.config)
            client_params = {k:v.to("cpu") for k, v in client_params.items()}
            client_size = len(client.train_loader)
            local_epochs = self.config.get('epochs', 1)
            steps = local_epochs * len(client.train_loader)

            client_sizes.append(client_size)
            client_steps.append(steps)

            # Calculate delta: (w_k - w_t)
            delta = {k: (client_params[k].to("cpu") - global_params[k]) for k in global_params}
            # g_k = tau_k * (w_k - w_t)
            scaled_delta = {k: v * steps for k, v in delta.items()}

            client_updates.append(scaled_delta)

        total_size = sum(client_sizes)
        aggregated_update = {
            k: torch.zeros_like(global_params[k]) for k in global_params
        }

        # Aggregate: Δw = ∑ (n_k / n_total) * (g_k / τ_k)
        for update, size, steps in zip(client_updates, client_sizes, client_steps):
            weight = torch.tensor(size / total_size)
            for k in aggregated_update:
                aggregated_update[k] = aggregated_update[k] + weight * (update[k] / steps)

        # Update global model
        new_global_params = {
            k: global_params[k] + aggregated_update[k] for k in global_params
        }

        return new_global_params
    
    def fit(self):
        global_params = self.global_model.state_dict()
        
        # Set global parameters to each client
        for client in self.clients:
            client.set_parameters(global_params)
        
        # Train each client
        fit_results = []
        for client in self.clients:
            fit_results.append(client.fit(global_params, self.config))
        
        # Update global model with FedNova
        self.update_global_model()
        
        # Aggregate the results
        aggregated_metrics = defaultdict(list)
        for result in fit_results:
            train_metrics = result[2]
            for met in train_metrics:
                aggregated_metrics[met].append(np.mean(train_metrics[met]))
        
        return aggregated_metrics
    
    def evaluate(self):
        global_params = self.global_model.state_dict()
        
        # Set global parameters to each client
        for client in self.clients:
            client.set_parameters(global_params)
        
        # Evaluate each client
        eval_results = []
        for client in self.clients:
            eval_results.append(client.evaluate(global_params, self.config))
        
        # Aggregate the results (validation metrics)
        aggregated_metrics = defaultdict(list)
        for result in eval_results:
            val_metrics = result[2]
            for met in val_metrics:
                aggregated_metrics[met].append(np.mean(val_metrics[met]))
        
        return aggregated_metrics