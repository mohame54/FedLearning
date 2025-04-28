from abc import ABC, abstractmethod
import torch
import numpy as np
from collections import defaultdict


class BaseFedStrategy(ABC):
    def __init__(self, clients, config, initial_parameters=None):
        """
        Initialize the base federated strategy.
        
        Parameters:
        - clients (list of GWNETClient): List of client instances.
        - config (dict): Configuration dictionary for federated learning.
        """
        self.clients = clients
        self.config = config
        self.initial_parameters = initial_parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = self.clients[0].model  # Assuming all clients have the same model structure

    @abstractmethod
    def aggregate_parameters(self):
        """
        Abstract method for parameter aggregation.
        This must be implemented by subclasses like FedAvg, FedAda, etc.
        """
        pass

    def fit(self):
        global_params = self.aggregate_parameters()

        for client in self.clients:
            client.set_parameters(global_params)

        fit_results = []
        for client in self.clients:
            try:
                result = client.fit(global_params, self.config)
                fit_results.append(result)
            except Exception as e:
                print(f"[Client {client.partition_id}] ❌ Fit crashed: {str(e)}")
                continue

        # Aggregate only successful clients
        self.update_global_model()
        aggregated_metrics = defaultdict(list)
        for _, _, train_metrics in fit_results:
            if "error" in train_metrics:
                print(f"[!] Skipping client with error: {train_metrics['error']}")
                continue
            for met in train_metrics:
                if isinstance(train_metrics[met], (int, float, np.ndarray)):
                    aggregated_metrics[met].append(np.mean(train_metrics[met]))
                else:
                    print(f"⚠️ Skipping non-numeric metric '{met}': {train_metrics[met]}")

        return aggregated_metrics

    def evaluate(self):
        global_params = self.aggregate_parameters()

        # Set global model weights to each client
        for client in self.clients:
            client.set_parameters(global_params)

        # Evaluate each client
        eval_results = []
        for client in self.clients:
            eval_results.append(client.evaluate(global_params, self.config))

        # Aggregate validation metrics
        aggregated_metrics = defaultdict(list)
        for result in eval_results:
            val_metrics = result[2]
            for met in val_metrics:
                aggregated_metrics[met].append(np.mean(val_metrics[met]))

        return aggregated_metrics

    def update_global_model(self):
        """
        Update the global model by setting the aggregated parameters.
        """
        aggregated_params = self.aggregate_parameters()
        self.initial_parameters = aggregated_params
        
        # Set the global model parameters
        self.global_model = self.global_model.to(self.device)
        self.global_model.load_state_dict(aggregated_params)
        self.global_model = self.global_model.to("cpu")

    def get_global_model(self):
        """
        Return the global model.
        """
        return self.global_model
    
    def set_parameters(self, parameters):
        for client in self.clients:
            client.set_parameters(parameters)
        self.global_model.load_state_dict(parameters)