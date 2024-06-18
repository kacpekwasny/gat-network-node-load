import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
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


def plot_losses(train_losses, val_losses, test_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    plt.grid(True)
    plt.savefig('loss_curves.png')
    plt.show()


def plot_additional_metrics(train_losses, val_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, test_losses, label='Test Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Test Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()


def analyze_results(train_losses, val_losses, test_losses):
    results = {
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Test Loss': test_losses
    }
    results_df = pd.DataFrame(results)
    print("Summary Statistics:\n")
    print(results_df.describe())

    plot_losses(train_losses, val_losses, test_losses)
    plot_additional_metrics(train_losses, val_losses, test_losses)


def train():
    data = torch.load('data.pt')

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    data.x = torch.tensor(scaler_x.fit_transform(data.x), dtype=torch.float)
    data.y = torch.tensor(scaler_y.fit_transform(
        data.y.reshape(-1, 1)), dtype=torch.float)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(num_node_features=data.num_features, num_classes=1).to(device)
    data = data.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=100, factor=0.5)

    best_val_loss = float('inf')
    patience = 250
    patience_counter = 0

    train_losses = []
    val_losses = []
    test_losses = []

    model.train()
    for epoch in range(2001):
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out[data.train_mask],
                          data.y[data.train_mask].view(-1, 1))
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        train_loss = F.mse_loss(
            out[data.train_mask], data.y[data.train_mask].view(-1, 1)).item()
        val_loss = F.mse_loss(out[data.val_mask],
                              data.y[data.val_mask].view(-1, 1)).item()
        test_loss = F.mse_loss(out[data.test_mask],
                               data.y[data.test_mask].view(-1, 1)).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 10 == 0:
            print(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}')

    analyze_results(train_losses, val_losses, test_losses)


if __name__ == "__main__":
    train()
