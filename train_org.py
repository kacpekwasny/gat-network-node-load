
import torch
import random
import pandas as pd
import numpy as np
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


FIRST_YEAR = 1995
LAST_YEAR = 2019
FEATURES = ['pop', 'cpi', 'emp']
NUM_TRAIN = 15
NUM_VAL = 3
NUM_TEST = 6
NUM_EDGE_FEATURES = 10
EDGE_FEATURES = ['f'+str(i) for i in range(NUM_EDGE_FEATURES)]

# The data is found in the project's Github.
DOWNLOAD_PREFIX = 'https://raw.githubusercontent.com/pboennig/gnns_for_gdp/master/'


def create_data(year):
    '''
    For given year, pull in node features, edge features, and edge index and
    save in a PyG Data object.
    '''

    assert (year in range(FIRST_YEAR, LAST_YEAR + 1))
    edges = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/X_EDGE_{year}.csv')

    # generate map from iso_code to ids of form [0, ..., num_unique_iso_codes - 1]
    iso_codes = set(edges['i'])
    iso_codes = iso_codes.union(set(edges['j']))
    iso_code_to_id = {code: i for (i, code) in enumerate(iso_codes)}

    # load in edge index
    edges['i_id'] = edges['i'].map(iso_code_to_id)
    edges['j_id'] = edges['j'].map(iso_code_to_id)
    edge_index = torch.from_numpy(
        edges[['i_id', 'j_id']].to_numpy(np.long)).t()
    # extract the features from the dataset.
    edge_attr = torch.from_numpy(edges[EDGE_FEATURES].to_numpy(np.float32))
    edge_attr = (edge_attr - edge_attr.mean(axis=0)) / (edge_attr.std(axis=0))

    # load in target values
    y_df = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/Y_{year}.csv')
    y_df['id'] = y_df['iso_code'].map(iso_code_to_id)
    y = torch.from_numpy(y_df.sort_values(
        'id')[f'{year+1}'].to_numpy(np.float32)).unsqueeze(1)  # get labels as tensor
    y = y.log()  # log scale since spread of GDP is large

    # load in input features
    x_df = pd.read_csv(f'{DOWNLOAD_PREFIX}/output/X_NODE_{year}.csv')
    x_df['id'] = x_df['iso_code'].map(iso_code_to_id)
    features = ['pop', 'cpi', 'emp']
    x = torch.from_numpy(x_df.sort_values(
        'id').loc[:, features].to_numpy(np.float32))
    x = (x - x.mean(axis=0)) / (x.std(axis=0))  # scale and center data
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def evaluate_model(model, data_iter):
    '''
    Accumulate MSE over a data list or loader.
    '''
    return sum([F.mse_loss(model(data), data.y).item() for data in data_iter])


def get_data():
    '''
    Generate data_lists for train, val, and test. These lists can be either loaded into data_loaders
    or indexed directly.
    '''

    data_list = [create_data(year) for year in range(FIRST_YEAR, LAST_YEAR)]
    random.shuffle(data_list)
    data_train = data_list[:NUM_TRAIN]
    data_val = data_list[NUM_TRAIN:NUM_TRAIN+NUM_VAL+1]
    data_test = data_list[NUM_TRAIN+NUM_VAL:]
    return (data_train, data_val, data_test)


# the function described above, these data are what we'll work with
data_train, data_val, data_test = get_data()

hyperparams = {
    'batch_size': 3,
    'save_loss_interval': 10,
    'print_interval': 50,
    'save_model_interval': 250,
    'n_epochs': 1500,
    'learning_rate': 0.01
}


def train(model, name_prefix, hyperparams):
    ''' 
    Train model with given hyperparams dict.

    Saves the following CSVs over the course of training:
    1. the loss trajectory: the val and train loss every save_loss_interval epochs at
       filename 'results/{name_prefix}_{learning_rate}_train.csv' e.g. 'results/baseline_0.05_train.csv'
    2. every save_model_interval save both the model at e.g. 'models/baseline_0.05_0_out_of_1000.pt`
       and the predicted values vs actual values in `results/baseline_0.05_0_out_of_1000_prediction.csv' on the test data.
    '''
    learning_rate = hyperparams['learning_rate']
    batch_size = hyperparams['batch_size']
    n_epochs = hyperparams['n_epochs']
    save_loss_interval = hyperparams['save_loss_interval']
    print_interval = hyperparams['print_interval']
    save_model_interval = hyperparams['save_model_interval']

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    losses = []
    test_data = data_test[0]
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
        if epoch % save_loss_interval == 0:
            val_loss = evaluate_model(model, data_val) / NUM_VAL
            train_loss = epoch_loss / NUM_TRAIN * batch_size
            if epoch % print_interval == 0:
                print("Epoch: {} Train loss: {:.2e} Validation loss: {:.2e}".format(
                    epoch, train_loss, val_loss))
            losses.append((epoch, train_loss, val_loss))
        if epoch % save_model_interval == 0:
            # save predictions for plotting
            model.eval()

    return losses
