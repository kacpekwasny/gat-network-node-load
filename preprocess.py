import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess(node_data: pd.DataFrame | None, edge_data: pd.DataFrame | None,
               save: bool, suffix: str) -> Data:

    if node_data is None:
        node_data = pd.read_csv('network_data.csv')

    if edge_data is None:
        edge_data = pd.read_csv('network_edges.csv')

    # Convert forward_history from string to list of integers
    node_data['forward_history'] = node_data['forward_history'].apply(
        lambda x: list(map(int, x.strip('[]').split(','))))

    # Generate features from forward_history
    scaler = MinMaxScaler()
    forward_history = scaler.fit_transform(
        np.array(node_data['forward_avg'].tolist()))

    generated_packets = scaler.fit_transform(
        np.array(node_data['generated_avg'].tolist()))
    generated_packets = generated_packets / max(generated_packets)

    # Use edge indices from the edge_data file
    edge_index = np.array([edge_data['source'], edge_data['target']])

    data = Data(
        x=torch.tensor(generated_packets, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.float),
        y=torch.tensor(forward_history, dtype=torch.float),
    )

    data.train_mask = torch.tensor(
        [i < 0.8 * len(node_data) for i in range(len(node_data))],
        dtype=torch.bool
    )
    data.val_mask = torch.tensor(
        [0.8 * len(node_data) <= i < 0.9 * len(node_data)
         for i in range(len(node_data))],
        dtype=torch.bool
    )
    data.test_mask = torch.tensor(
        [i >= 0.9 * len(node_data) for i in range(len(node_data))],
        dtype=torch.bool
    )

    if save:
        torch.save(data, f'data_{suffix}.pt')

    return data


if __name__ == "__main__":
    data = preprocess(False, "")
