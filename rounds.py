from client import GWNETClient
from utils import DataCalss, prepare_data, get_adj_mets, write_json
from strategies.feds import FedAvg, FedProx, FedNova

config = DataCalss.from_yaml_file("config.yaml")

adj, _ = get_adj_mets(config.adj_file_pth)
train_loaders, val_loaders, test_loader = prepare_data(
    config.dataset_folder_pth,
    config.num_clients,
    config.train_batch_size,
    config.val_batch_size,
    config.seed,
    **config.loader_kwargs.as_dict()
)
adj, _ = get_adj_mets(config.adj_file_pth)

clients = [GWNETClient(partition_id=i, adj_met=adj, model_config=config.model_config.as_dict(), 
                       train_loader=train_loaders[i], val_loader=val_loaders[i]) for i in range( config.num_clients)]

# Initialize the strategy (FedAvg)

fed_avg_strategy = FedAvg(clients, config.config_fit.as_dict()) # 
result_path = config.config_fit.results.test_pth
data = {"train":[], "val":[]}

for i in range(config.num_rounds):
    print(f"Round {i + 1}\{config.num_rounds}")
    train_results = fed_avg_strategy.fit()
    data['train'].append(train_results)

    # Evaluate the global model
    eval_results = fed_avg_strategy.evaluate()
    data['val'].append(eval_results)


    # Get the updated global model
    fed_avg_strategy.update_global_model()
    global_model = fed_avg_strategy.get_global_model()

write_json(result_path, data)