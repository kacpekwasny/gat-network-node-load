import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
from urllib.request import urlopen
from zipfile import ZipFile, BadZipFile
from io import BytesIO
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, temporal_signal_split
from torch_geometric_temporal.nn.recurrent import A3TGCN

# Seed for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the path to the folder containing the files
folder_path = 'PeMSD7_Full'

# Check if the files exist
if not os.path.exists(os.path.join(folder_path, 'PeMSD7_V_228.csv')) or not os.path.exists(os.path.join(folder_path, 'PeMSD7_W_228.csv')):
    print("Required CSV files are missing. Please check the folder path and ensure the files are present.")
    exit()

# Load dataset
speeds = pd.read_csv(os.path.join(folder_path, 'PeMSD7_V_228.csv'), names=range(0, 228))
distances = pd.read_csv(os.path.join(folder_path, 'PeMSD7_W_228.csv'), names=range(0, 228))

# Plot traffic speeds
def plot_speeds(speeds):
    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(speeds)
    plt.grid(linestyle=':')
    plt.xlabel('Time (5 min)')
    plt.ylabel('Traffic speed')
    plt.show()

# Plot mean/std traffic speed
def plot_mean_std(speeds):
    mean = speeds.mean(axis=1)
    std = speeds.std(axis=1)

    plt.figure(figsize=(10, 5), dpi=100)
    plt.plot(mean, 'k-')
    plt.grid(linestyle=':')
    plt.fill_between(mean.index, mean - std, mean + std, color='r', alpha=0.1)
    plt.xlabel('Time (5 min)')
    plt.ylabel('Traffic speed')
    plt.show()

# Plot distance and correlation matrices
def plot_matrices(distances, speeds):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    fig.tight_layout(pad=3.0)
    ax1.matshow(distances)
    ax1.set_xlabel("Sensor station")
    ax1.set_ylabel("Sensor station")
    ax1.title.set_text("Distance matrix")
    ax2.matshow(-np.corrcoef(speeds.T))
    ax2.set_xlabel("Sensor station")
    ax2.set_ylabel("Sensor station")
    ax2.title.set_text("Correlation matrix")
    plt.show()

# Compute adjacency matrix
def compute_adj(distances, sigma2=0.1, epsilon=0.5):
    d = distances.to_numpy() / 10000.
    d2 = d * d
    n = distances.shape[0]
    w_mask = np.ones([n, n]) - np.identity(n)
    return np.exp(-d2 / sigma2) * (np.exp(-d2 / sigma2) >= epsilon) * w_mask

adj = compute_adj(distances)

# Plot adjacency matrix
def plot_adj(adj):
    plt.figure(figsize=(8, 8))
    cax = plt.matshow(adj, False)
    plt.colorbar(cax)
    plt.xlabel("Sensor station")
    plt.ylabel("Sensor station")
    plt.show()

# Plot graph
def plot_graph(adj):
    plt.figure(figsize=(10, 5))
    rows, cols = np.where(adj > 0)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    G.add_edges_from(edges)
    nx.draw(G, with_labels=True)
    plt.show()

# Z-score normalization
def zscore(x, mean, std):
    return (x - mean) / std

speeds_norm = zscore(speeds, speeds.mean(axis=0), speeds.std(axis=0))

# Create dataset
lags = 24
horizon = 48
xs = []
ys = []
for i in range(lags, speeds_norm.shape[0] - horizon):
    xs.append(speeds_norm.to_numpy()[i - lags:i].T)
    ys.append(speeds_norm.to_numpy()[i + horizon - 1])

# Convert adjacency matrix to edge_index (COO format)
edge_index = (np.array(adj) > 0).nonzero()

dataset = StaticGraphTemporalSignal(edge_index, adj[adj > 0], xs, ys)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

# Implementing the A3T-GCN architecture
class TemporalGNN(torch.nn.Module):
    def __init__(self, dim_in, periods):
        super().__init__()
        self.tgnn = A3TGCN(in_channels=dim_in, out_channels=32, periods=periods)
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index, edge_attr):
        h = self.tgnn(x, edge_index, edge_attr).relu()
        h = self.linear(h)
        return h

model = TemporalGNN(lags, 1).to('cpu')
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
model.train()
print(model)

# Training
for epoch in range(30):
    loss = 0
    step = 0
    for i, snapshot in enumerate(train_dataset):
        y_pred = model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_attr)
        loss += torch.mean((y_pred - snapshot.y) ** 2)
        step += 1
    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:>2} | Train MSE: {loss:.4f}")

# Inverse z-score normalization
def inverse_zscore(x, mean, std):
    return x * std + mean

y_test = []
for snapshot in test_dataset:
    y_hat = snapshot.y.numpy()
    y_hat = inverse_zscore(y_hat, speeds.mean(axis=0), speeds.std(axis=0))
    y_test = np.append(y_test, y_hat)

gnn_pred = []
model.eval()
for snapshot in test_dataset:
    snapshot = snapshot
    y_hat = model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_weight).squeeze().detach().numpy()
    y_hat = inverse_zscore(y_hat, speeds.mean(axis=0), speeds.std(axis=0))
    gnn_pred = np.append(gnn_pred, y_hat)

def MAE(real, pred):
    return np.mean(np.abs(pred - real))

def RMSE(real, pred):
    return np.sqrt(np.mean((pred - real) ** 2))

def MAPE(real, pred):
    return np.mean(np.abs(pred - real) / (real + 1e-5))

print(f'GNN MAE  = {MAE(gnn_pred, y_test):.4f}')
print(f'GNN RMSE = {RMSE(gnn_pred, y_test):.4f}')
print(f'GNN MAPE = {MAPE(gnn_pred, y_test):.4f}')

rw_pred = []
for snapshot in test_dataset:
    y_hat = snapshot.x[:, -1].squeeze().detach().numpy()
    y_hat = inverse_zscore(y_hat, speeds.mean(axis=0), speeds.std(axis=0))
    rw_pred = np.append(rw_pred, y_hat)

print(f'RW MAE  = {MAE(rw_pred, y_test):.4f}')
print(f'RW RMSE = {RMSE(rw_pred, y_test):.4f}')
print(f'RW MAPE = {MAPE(rw_pred, y_test):.4f}')

ha_pred = []
for i in range(lags, speeds_norm.shape[0] - horizon):
    y_hat = speeds_norm.to_numpy()[:i].T.mean(axis=1)
    y_hat = inverse_zscore(y_hat, speeds.mean(axis=0), speeds.std(axis=0))
    ha_pred.append(y_hat)
ha_pred = np.array(ha_pred).flatten()[-len(y_test):]

print(f'HA MAE  = {MAE(ha_pred, y_test):.4f}')
print(f'HA RMSE = {RMSE(ha_pred, y_test):.4f}')
print(f'HA MAPE = {MAPE(ha_pred, y_test):.4f}')

y_preds = [inverse_zscore(model(snapshot.x.unsqueeze(2), snapshot.edge_index, snapshot.edge_weight).squeeze().detach().numpy(), speeds.mean(axis=0), speeds.std(axis=0)).mean() for snapshot in test_dataset]

mean = speeds.mean(axis=1)
std = speeds.std(axis=1)

plt.figure(figsize=(10, 5))
plt.plot(np.array(mean), 'k-', label='Mean')
plt.plot(range(len(speeds) - len(y_preds), len(speeds)), y_preds, 'r-', label='Prediction')
plt.grid(linestyle=':')
plt.fill_between(mean.index, mean - std, mean + std, color='r', alpha=0.1)
plt.axvline(x=len(speeds) - len(y_preds), color='b', linestyle='--')
plt.xlabel('Time (5 min)')
plt.ylabel('Traffic speed to predict')
plt.legend(loc='upper right')
plt.show()
