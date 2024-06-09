import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import GATConv, BatchNorm

NUM_EDGE_FEATURES = 32


class GAT(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.convs = [
            GATConv(self.num_features, self.hidden_size,
                    edge_dim=NUM_EDGE_FEATURES),
            GATConv(self.hidden_size, self.hidden_size,
                    edge_dim=NUM_EDGE_FEATURES)
        ]
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            # adding edge features here!
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr=edge_attr)
        x = self.linear(x)

        return F.relu(x)


class GNN(torch.nn.Module):
    def __init__(self, num_features=3, hidden_size=32, target_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_features = num_features
        self.target_size = target_size
        self.convs = [GATConv(self.num_features, self.hidden_size),
                      GATConv(self.hidden_size, self.hidden_size)]
        self.linear = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.convs[-1](x, edge_index)
        x = self.linear(x)
        return F.relu(x)  # since we know Y = log_gdp > 0, enforce via relu


class MyGAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(MyGAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 32, heads=8, dropout=0.2)
        self.bn1 = BatchNorm(32 * 8)
        self.conv2 = GATConv(32 * 8, 16, heads=8, dropout=0.2)
        self.bn2 = BatchNorm(16 * 8)
        self.conv3 = GATConv(16 * 8, 8, heads=8, dropout=0.2)
        self.bn3 = BatchNorm(8 * 8)
        self.conv4 = GATConv(8 * 8, num_classes, heads=1,
                             concat=False, dropout=0.2)
        self.bn4 = BatchNorm(num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        return x
