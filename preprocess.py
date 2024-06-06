import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess():
    node_data = pd.read_csv('network_data.csv')
    edge_data = pd.read_csv('network_edges.csv')

    # Convert forward_history from string to list of integers
    node_data['forward_history'] = node_data['forward_history'].apply(lambda x: list(map(int, x.strip('[]').split(','))))

    # Generate features from forward_history
    scaler = MinMaxScaler()
    node_features = scaler.fit_transform(np.array(node_data['forward_history'].tolist()))
    
    # Use edge indices from the edge_data file
    edge_index = np.array([edge_data['source'], edge_data['target']])

    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(node_data.index, dtype=torch.float),  # Assuming node labels are node indices
    )

    data.train_mask = torch.tensor([True if i < 0.8 * len(node_data) else False for i in range(len(node_data))], dtype=torch.bool)
    data.val_mask = torch.tensor([True if 0.8 * len(node_data) <= i < 0.9 * len(node_data) else False for i in range(len(node_data))], dtype=torch.bool)
    data.test_mask = torch.tensor([True if i >= 0.9 * len(node_data) else False for i in range(len(node_data))], dtype=torch.bool)

    torch.save(data, 'data.pt')

if __name__ == "__main__":
    preprocess()
