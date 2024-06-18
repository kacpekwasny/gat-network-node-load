from datetime import datetime
import json
import torch
import random
import torch.nn.functional as F
import pandas as pd
import numpy as np

from pathlib import Path

from torch_geometric.data import Data, DataLoader

import old_ver.models as models
from data_generate import DATA_DIR

from data_read import data_read_dir



def evaluate_model(model, data_iter):
    '''
    Accumulate MSE over a data list or loader.
    '''
    return sum([F.mse_loss(model(data), data.y).item() for data in data_iter])


def get_data() -> tuple[list[Data], list[Data]]:
    '''
    Generate data_lists for train, val, and test. These lists can be either loaded into data_loaders
    or indexed directly.
    '''
    num_train = NUM_TRAIN

    data_list: list[Data] = [read_data(i) for i, _ in enumerate(DATA_DIR.glob("EDGE_*"))]
    random.shuffle(data_list)

    idx_train = num_train * len(data_list) // 100
    if idx_train == 0:
        idx_train = 1
    data_train = data_list[:idx_train]
    data_val = data_list[idx_train:]
    return (data_train, data_val)



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
            val_loss = evaluate_model(model, data_val) / (100 - NUM_TRAIN) 
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


#########################################
# 80% of data is train, 20% is validate #
NUM_TRAIN = 80
NODES_NUM = 5
MODEL_DIR = Path(__file__).parent / "models"

model = models.GNNKacper(num_features=6, hidden_size=50, target_size=1)
# model = models.TemporalGNN(dim_in=6, periods=1)

if __name__ == "__main__":
    from sys import argv 
    if len(argv) > 1:
        DATA_DIR = DATA_DIR.parent / argv[1]
        
    # the function described above, these data are what we'll work with
    data_train, data_val = get_data()

    hyperparams = {
        'batch_size': 256,
        'save_loss_interval': 10,
        'print_interval': 50,
        'save_model_interval': 250,
        'n_epochs': 1500,
        'learning_rate': 0.001
    }
    model, losses = train(model)

    import old_ver.analyze as az
    # az.comparison_plot(losses)
    
    # az.draw_results(d.edge_index.T, model(d).detach(), d.y)



