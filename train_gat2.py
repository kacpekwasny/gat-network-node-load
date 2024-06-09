from datetime import datetime
import json
import torch
import random
import torch.nn.functional as F
import pandas as pd
import numpy as np

from pathlib import Path

from torch_geometric.data import Data, DataLoader

import models
from generate_network_data import DATA_DIR

MODEL_DIR = Path(__file__).parent / "models"

NUM_TRAIN = 80
NUM_VAL = 10
NUM_TEST = 10


def read_data(i: int):
    '''
    For given year, pull in node features, edge features, and edge index and
    save in a PyG Data object.
    '''

    node = pd.read_csv(DATA_DIR / f"NODE_{i}.csv")
    edge = pd.read_csv(DATA_DIR / f"EDGE_{i}.csv")

    edge_index = torch.from_numpy(
        edge[['source', 'target']].to_numpy(np.longlong)).t()

    # if we would implement some edge features, use those lines
    # edge_attr = torch.from_numpy(edges.to_numpy(np.float32))
    # edge_attr = (edge_attr - edge_attr.mean(axis=0)) / (edge_attr.std(axis=0))

    node['routing'] = node['routing'].map(json.loads).map(list)
    node_x_gen_packets = torch.from_numpy(
            node['generated_packets_avg'].to_numpy(np.float32)[np.newaxis, :]) # .unsqueeze(1)
    node_x_routing = torch.from_numpy(np.array(node['routing'].tolist(), dtype=np.float32))
    node_x = torch.concat((node_x_gen_packets, node_x_routing))
    node_x = (node_x - node_x.flatten().min()) / node_x.flatten().max()

    node_y = torch.from_numpy(
        node['forward_avg'].to_numpy(np.float32)) # .unsqueeze(1)
    node_y = node_y / node_y.flatten().max()

    return Data(x=node_x, edge_index=edge_index, y=node_y)


def evaluate_model(model, data_iter):
    '''
    Accumulate MSE over a data list or loader.
    '''
    return sum([F.mse_loss(model(data), data.y).item() for data in data_iter])


def get_data(number_data: int, num_train: int, num_val: int) -> tuple[list[Data], list[Data], list[Data]]:
    '''
    Generate data_lists for train, val, and test. These lists can be either loaded into data_loaders
    or indexed directly.
    '''

    data_list: list[Data] = [read_data(i) for i in range(number_data)]
    random.shuffle(data_list)
    data_train = data_list[:num_train]
    data_val = data_list[num_train:num_train+num_val]
    data_test = data_list[num_train+num_val:]
    return (data_train, data_val, data_test)


# the function described above, these data are what we'll work with
data_train, data_val, data_test = get_data(100, NUM_TRAIN, NUM_VAL)

hyperparams = {
    'batch_size': 8,
    'save_loss_interval': 10,
    'print_interval': 50,
    'save_model_interval': 250,
    'n_epochs': 1500,
    'learning_rate': 0.0001
}

def train(model: torch.nn.Module):
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_epochs = hyperparams['n_epochs']
    save_loss_interval = hyperparams['save_loss_interval']
    print_interval = hyperparams['print_interval']
    save_model_interval = hyperparams['save_model_interval']

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        model.train()
        for data in loader:
            optimizer.zero_grad()
            out = model(data)
            loss = F.mse_loss(out, data.y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        if  (epoch % save_loss_interval == 0):
            val_loss = evaluate_model(model, data_val) / NUM_VAL
            train_loss = epoch_loss / NUM_TRAIN * batch_size
            if (epoch < 100) or (epoch % print_interval == 0):
                print("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}".format(
                    epoch, train_loss, val_loss))
            losses.append((epoch, train_loss, val_loss))
        if epoch % save_model_interval == 0:
            # save predictions for plotting
            model.eval()

    torch.save(model.state_dict(), MODEL_DIR / f"MODEL_{datetime.now().isoformat()}_{val_loss}.pt")

    return model, losses


if __name__ == "__main__":
    model, losses = train(models.GNNKacper(num_features=50, hidden_size=128))

    import analyze as az
    az.comparison_plot(losses)
    d = data_test[0]
    
    # az.draw_results(d.edge_index.T, model(d).detach(), d.y)



