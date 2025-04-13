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


class FedAdam(BaseFedStrategy):
    def __init__(self, clients, config):
        super().__init__(clients, config)
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (velocity)
        self.tau = 0  # Time step
        
        # Initialize momentum and velocity with zeros for each parameter
        dummy_params = self.clients[0].get_parameters(self.config)
        for name, param in dummy_params.items():
            self.m[name] = torch.zeros_like(param)
            self.v[name] = torch.zeros_like(param)
    
    def aggregate_parameters(self):
        client_parameters = []
        client_sizes = []
        
        for client in self.clients:
            client_parameters.append(client.get_parameters(self.config))
            client_sizes.append(len(client.train_loader))
        
        total_size = sum(client_sizes)
        
        # Compute weighted average of client parameters (delta)
        delta = {}
        for param_name in client_parameters[0].keys():
            weighted_sum = torch.zeros_like(client_parameters[0][param_name])
            for client_params, size in zip(client_parameters, client_sizes):
                weighted_sum += (client_params[param_name] * size / total_size)
            delta[param_name] = weighted_sum
        
        # Get Adam hyperparameters from config
        beta1 = self.config.get('beta1', 0.9)
        beta2 = self.config.get('beta2', 0.999)
        eta = self.config.get('eta', 0.01)  # Learning rate
        epsilon = self.config.get('epsilon', 1e-8)  # Small constant for numerical stability
        
        # Apply Adam update rule
        self.tau += 1
        aggregated_params = {}
        
        current_global_params = self.global_model.state_dict()
        
        for param_name in delta.keys():
            # Update momentum and velocity
            self.m[param_name] = beta1 * self.m[param_name] + (1 - beta1) * delta[param_name]
            self.v[param_name] = beta2 * self.v[param_name] + (1 - beta2) * (delta[param_name] ** 2)
            
            # Bias correction
            m_hat = self.m[param_name] / (1 - beta1 ** self.tau)
            v_hat = self.v[param_name] / (1 - beta2 ** self.tau)
            
            # Update global model parameters
            aggregated_params[param_name] = current_global_params[param_name] + eta * m_hat / (torch.sqrt(v_hat) + epsilon)
        
        return aggregated_params
    
    def fit(self):
        # Update global model with Adam
        self.update_global_model()
        global_params = self.global_model.state_dict()
        
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


class FedProx(BaseFedStrategy):
    def __init__(self, clients, config):
        super().__init__(clients, config)
        # Proximal term coefficient (mu)
        self.mu = config.get('mu', 0.01)
    
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
                aggregated_params[param_name] += (client_params[param_name] * size / total_size)
        
        return aggregated_params
    
    def fit(self):
        global_params = self.aggregate_parameters()
        
        # Set global parameters to each client
        for client in self.clients:
            client.set_parameters(global_params)
        
        # Train each client with proximal term
        fit_results = []
        for client in self.clients:
            # Add proximal term to client's configuration
            client_config = self.config.copy()
            client_config['global_params'] = global_params
            client_config['mu'] = self.mu
            
            # Custom fit method for FedProx that includes proximal term in loss
            fit_results.append(self.proximal_fit(client, global_params, client_config))
        
        # Aggregate the results
        aggregated_metrics = defaultdict(list)
        for result in fit_results:
            train_metrics = result[2]
            for met in train_metrics:
                aggregated_metrics[met].append(np.mean(train_metrics[met]))
        
        return aggregated_metrics
    
    def proximal_fit(self, client, global_params, config):
        """
        Custom fit method for FedProx that adds a proximal term to the loss function.
        """
        # Get original fit logic from client but modify training loop to include proximal term
        local_model = client.model
        local_model.train()
        
        optimizer = client.get_optimizer(config)
        criterion = client.get_criterion(config)
        
        train_loss = []
        train_metrics = defaultdict(list)
        
        for epoch in range(config['epochs']):
            epoch_loss = []
            epoch_metrics = defaultdict(list)
            
            for batch_idx, (data, target) in enumerate(client.train_loader):
                data, target = data.to(client.device), target.to(client.device)
                
                optimizer.zero_grad()
                output = local_model(data)
                
                # Original task loss
                loss = criterion(output, target)
                
                # Add proximal term to loss
                proximal_term = 0.0
                for name, param in local_model.named_parameters():
                    proximal_term += torch.sum((param - global_params[name].to(client.device))**2)
                
                # Update loss with proximal term
                loss += (config['mu'] / 2) * proximal_term
                
                loss.backward()
                optimizer.step()
                
                epoch_loss.append(loss.item())
                metrics = client.compute_metrics(output, target)
                
                for met in metrics:
                    epoch_metrics[met].append(metrics[met])
            
            train_loss.append(np.mean(epoch_loss))
            for met in epoch_metrics:
                train_metrics[met].append(np.mean(epoch_metrics[met]))
        
        return local_model.state_dict(), train_loss, train_metrics
    
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
        client_parameters = []
        client_sizes = []
        client_steps = []  # Local steps (tau) for each client
        
        for client in self.clients:
            client_parameters.append(client.get_parameters(self.config))
            client_sizes.append(len(client.train_loader))
            
            # Calculate effective number of local steps
            # This is the number of local epochs * number of batches per epoch
            local_epochs = self.config.get('epochs', 1)
            batches_per_epoch = len(client.train_loader)
            client_steps.append(local_epochs * batches_per_epoch)
        
        total_size = sum(client_sizes)
        aggregated_params = {}
        
        # Calculate normalized weights for FedNova aggregation
        # Weight = (client_size * client_steps) / (total_clients * total_size)
        total_steps_weighted = sum(size * steps for size, steps in zip(client_sizes, client_steps))
        normalized_weights = []
        
        for size, steps in zip(client_sizes, client_steps):
            if total_steps_weighted > 0:
                weight = (size * steps) / total_steps_weighted
            else:
                weight = 1.0 / len(self.clients)  # Fall back to equal weights
            normalized_weights.append(weight)
        
        # Retrieve current global model parameters
        current_global_params = self.global_model.state_dict()
        
        # Calculate normalized model updates using the normalized weights
        for param_name in client_parameters[0].keys():
            aggregated_params[param_name] = current_global_params[param_name].clone()
            
            # Apply normalized updates
            for client_params, weight, steps in zip(client_parameters, normalized_weights, client_steps):
                # Calculate the normalized update
                normalized_update = (client_params[param_name] - current_global_params[param_name]) * (weight / steps)
                aggregated_params[param_name] += normalized_update * sum(client_steps)
        
        return aggregated_params
    
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