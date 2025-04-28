import torch
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
    

class FedAdam(BaseFedStrategy):
    def __init__(self, clients, config):
        super().__init__(clients, config)

        # Parse FedAdam-specific optimizer config
        fedadam_cfg = config.get("fedadam_config", {})
        self.lr = fedadam_cfg.get("lr", 0.001)
        self.betas = (fedadam_cfg.get("beta1", 0.9), fedadam_cfg.get("beta2", 0.999))
        self.tau = float(fedadam_cfg.get("epsilon", 1e-8))
        self.v0 = fedadam_cfg.get("v0", 0.0)

        self.m = {}  # First moment estimate
        self.v = {}  # Second moment estimate

    def aggregate_parameters(self):
        client_parameters = []
        client_sizes = []

        for client in self.clients:
            client_parameters.append(client.get_parameters(self.config))
            client_sizes.append(len(client.train_loader))

        total_size = sum(client_sizes)
        aggregated_deltas = {}

        # Use first client's parameters as reference (for global params)
        global_params = self.clients[0].model.state_dict()

        for param_name in client_parameters[0].keys():
            delta = torch.zeros_like(client_parameters[0][param_name])
            if delta.dtype in [torch.int32, torch.int64]:
                delta = delta.float()
            global_param = self.clients[0].model.state_dict()[param_name]

            # Aggregate deltas weighted by client data size
            for client_params, size in zip(client_parameters, client_sizes):
                local_param = client_params[param_name]
                delta += (local_param - global_param) * (size / total_size)

            # Initialize m and v
            if param_name not in self.m:
                self.m[param_name] = torch.zeros_like(delta)
                self.v[param_name] = self.v0 * self.betas[1] + (1 - self.betas[1]) * delta.pow(2)

            # Update biased first moment estimate
            self.m[param_name] = self.betas[0] * self.m[param_name] + (1 - self.betas[0]) * delta

            # Update biased second raw moment estimate
            self.v[param_name] = self.betas[1] * self.v[param_name] + (1 - self.betas[1]) * delta.pow(2)

            # Apply FedAdam update rule
            adaptive_update = self.lr * self.m[param_name] / (self.v[param_name].sqrt() + self.tau)
            aggregated_deltas[param_name] = global_params[param_name] + adaptive_update

        return aggregated_deltas


class FedAvgM(BaseFedStrategy):
    def __init__(self, clients, config):
        super().__init__(clients, config)
        fedavgm_cfg = config.get("fedavgm_config", {})
        self.lr = fedavgm_cfg.get("lr", 0.01)
        self.server_momentum = fedavgm_cfg.get("momentum", 0.9)
        
        self.velocity = {}

    def aggregate_parameters(self):
        client_parameters = []
        client_sizes = []

        for client in self.clients:
            client_parameters.append(client.get_parameters(self.config))
            client_sizes.append(len(client.train_loader))

        total_size = sum(client_sizes)
        aggregated_deltas = {}

        # Use first client's parameters as reference (for global params)
        global_params = self.clients[0].model.state_dict()

        for param_name in client_parameters[0].keys():
            delta = torch.zeros_like(client_parameters[0][param_name])
            if delta.dtype in [torch.int32, torch.int64]:
                delta = delta.float()

            # Aggregate deltas weighted by client data size
            for client_params, size in zip(client_parameters, client_sizes):
                local_param = client_params[param_name]
                delta += (local_param - global_params[param_name]) * (size / total_size)

            # Initialize velocity if not present
            if param_name not in self.velocity:
                self.velocity[param_name] = torch.zeros_like(delta)

            # Update velocity with momentum
            self.velocity[param_name] = (self.server_momentum * self.velocity[param_name] + 
                                        delta)
            
            # Apply update with learning rate
            aggregated_deltas[param_name] = (global_params[param_name] + 
                                            self.lr * self.velocity[param_name])

        return aggregated_deltas