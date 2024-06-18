import pandas as pd
import numpy as np
import os
import json
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models_traffic_forecast import LSTMGC, GraphConv, GraphInfo
from data_read import data_read_dir, create_tf_dataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


def graph_info(adj: np.ndarray) -> GraphInfo:
    node_indices, neighbor_indices = np.where(adj == 1)
    return GraphInfo(
        edges=(node_indices.tolist(), neighbor_indices.tolist()),
        num_nodes=adj.shape[0],
    )


def train(dtrain: tf.data.Dataset, dval: tf.data.Dataset, nodes: np.ndarray, adj: np.ndarray):

    graph = graph_info(adj)

    print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

    st_gcn = LSTMGC(
        in_feat,
        out_feat,
        lstm_units,
        input_sequence_length,
        forecast_horizon,
        graph,
        graph_conv_params,
    )

    inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
    outputs = st_gcn(inputs)

    model = keras.models.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.002),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.RootMeanSquaredError()],
    )

    history = model.fit(
        dtrain,
        validation_data=dval,
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=10)],
    )

    # print(f"Training history: {history.history}")

    plot_training_history(history)

    return model


def plot_forward_history(node_data):
    for i, node in node_data.iterrows():
        forward_history = json.loads(node['forward_history'])
        plt.plot(forward_history, label=f'Node {node["node"]}')
    plt.xlabel('Ticks')
    plt.ylabel('Forward History')
    plt.title('Forward History over Time')
    plt.legend()
    plt.show()


def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Train MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Model Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend(loc='upper right')
    
    plt.show()


in_feat = 1
batch_size = 64
epochs = 20
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False
out_feat = 10
lstm_units = 64
graph_conv_params = {
    "aggregation_type": "mean",
    "combination_type": "concat",
    "activation": None,
}

if __name__ == "__main__":
    from sys import argv

    data = data_read_dir(argv[1])

    scaler = StandardScaler()
    for nodes, adj in data:
        steps = nodes.shape[0]

        print(nodes.shape, adj.shape)

        tn = nodes[:int(steps * 0.8)]
        vn = nodes[int(steps*0.8):]

        # Normalizacja danych treningowych i walidacyjnych
        tn = scaler.fit_transform(tn)
        vn = scaler.transform(vn)

        dt = create_tf_dataset(
            tn,
            input_sequence_length=input_sequence_length,
            forecast_horizon=forecast_horizon,
        )
        dv = create_tf_dataset(
            vn,
            input_sequence_length=input_sequence_length,
            forecast_horizon=forecast_horizon,
        )
        model = train(dt, dv, nodes, adj)

        # Wizualizacja historii przesyłu pakietów
        node_data = pd.read_csv(os.path.join(argv[1], 'NODE_0.csv'))
        plot_forward_history(node_data)
