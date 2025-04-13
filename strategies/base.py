from abc import ABC, abstractmethod
import torch


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

    @abstractmethod
    def fit(self):
        """
        Abstract method for one round of federated learning.
        This must be implemented by subclasses like FedAvg, FedAda, etc.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Abstract method for evaluating the global model on clients' validation datasets.
        This must be implemented by subclasses like FedAvg, FedAda, etc.
        """
        pass

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