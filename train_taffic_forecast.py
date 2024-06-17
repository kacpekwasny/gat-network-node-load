import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from models_traffic_forecast import LSTMGC, GraphConv, GraphInfo
from data_read import data_read_dir, create_tf_dataset




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
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
        loss=keras.losses.MeanSquaredError(),
    )

    model.fit(
        dtrain,
        validation_data=dval,
        epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(patience=10)],
    )


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
    for nodes, adj in data:
        steps = nodes.shape[0]

        tn = nodes[:int(steps * 0.8)]
        vn = nodes[int(steps*0.8):]
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
        train(dt, dv, nodes, adj)
