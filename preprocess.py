import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess():
    node_data = pd.read_csv('network_data.csv')
    edge_data = pd.read_csv('network_edges.csv')

    # Convert forward_history from string to list of integers
    node_data['forward_history'] = node_data['forward_history'].apply(
        lambda x: list(map(int, x.strip('[]').split(','))))

    # Generate features from forward_history
    scaler = MinMaxScaler()
    node_features = scaler.fit_transform(
        np.array(node_data['forward_history'].tolist()))

    # Use edge indices from the edge_data file
    edge_index = np.array([edge_data['source'], edge_data['target']])

    data = Data(
        # TODO - wczesniej przy generowaniu ruchu zrobic tak, zebysmy zapisywali
        # rowniez ile dane urzadzenie wygenerowalo pakietow (chyba to jest
        # parametr k), tutaj bysmy wtedy
        # te pakiety usredniali (preprocessing danych) i by bylo, ze ten wyslal
        # 0.1 pakietow, a ten wyslal 2 pakietow (po prostu wartosc wzgledna),
        # ale jak wiemy z wykladow to ML woli liczby male, mozna ilsoc
        # wygenerowanych pakietow potraktowac tez logarytmem
        x=torch.tensor(node_features, dtype=torch.float),

        edge_index=torch.tensor(edge_index, dtype=torch.long),

        # TODO, tutaj za to bierzemy historie forwardowania i liczymy srednia z
        # forwarding_history no i to jest wtedy takie obciazenie,
        # ktore router wiemy ze musi uciagnac w takiej sieci, przy takim
        # generowanym ruchu (taki jest usecase)
        # Assuming node labels are node indices
        y=torch.tensor(node_data.index, dtype=torch.float),
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

    torch.save(data, 'data.pt')


if __name__ == "__main__":
    preprocess()
