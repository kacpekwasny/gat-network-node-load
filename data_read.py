import json
from pathlib import Path

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

import torch
import pandas as pd
import tensorflow as tf
import numpy as np


def data_read_pair(data_dir: Path, i: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For given year, pull in node features, edge features, and edge index and
    save in a PyG Data object.
    """

    node = pd.read_csv(data_dir / f"NODE_{i}.csv")
    edge = pd.read_csv(data_dir / f"EDGE_{i}.csv")

    if "forward_history" in node:
        node["forward_history"] = node["forward_history"].map(json.loads)

    if "routing" in node:
        node["routing"] = node["routing"].map(json.loads)

    return node, edge


def data_process_node(df: pd.DataFrame) -> torch.Tensor:
    # if we would implement some edge features, use those lines
    # edge_attr = torch.from_numpy(edges.to_numpy(np.float32))
    # edge_attr = (edge_attr - edge_attr.mean(axis=0)) / (edge_attr.std(axis=0))

    node_x_gen_packets = torch.from_numpy(
        df["generated_packets_avg"].to_numpy(np.float32)[np.newaxis, :]
    )  # .unsqueeze(1)

    node_x_routing = torch.from_numpy(
        np.array(df["routing"].tolist(), dtype=np.float32)
    )

    node_x = torch.concat(
        (
            node_x_gen_packets.permute(1, 0),
            node_x_routing.reshape((NODES_NUM, NODES_NUM)),
        ),
        dim=1,
    )
    node_x = (node_x - node_x.flatten().min()) / node_x.flatten().max()

    node_y = torch.from_numpy(df["forward_avg"].to_numpy(np.float32))  # .unsqueeze(1)
    node_y = node_y - node_y.flatten().min()
    node_y = (node_y / node_y.flatten().max())[:, None]

    return Data(x=node_x, edge_index=edge_index, y=node_y)


def data_process_node_for_forward_history(
    df: pd.DataFrame,
) -> tuple[list[int], np.ndarray]:
    forward_his = df.sort_values(by="node")["forward_history"]
    forward_his = np.array(forward_his.to_list(), dtype=np.float32)
    forward_his = forward_his.transpose()

    mean = forward_his.mean(axis=0)
    std = forward_his.std(axis=0)

    forward_his = (forward_his - mean) / std
    forward_his = np.nan_to_num(forward_his)

    return forward_his


def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()


def data_process_edge(df: pd.DataFrame) -> torch.Tensor:
    # edge index
    eidx = torch.from_numpy(df[["source", "target"]].to_numpy(np.longlong)).t()
    eidx = np.array(eidx)
    max_idx = eidx.flatten().max(axis=None)

    adj = np.zeros((max_idx + 1, max_idx + 1))
    adj[eidx[0, :], eidx[1, :]] = 1

    return adj


def data_read_dir(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
        
    data_list: list[tuple[pd.DataFrame, pd.DataFrame]] = [
        data_read_pair(data_dir, i) for i, _ in enumerate(data_dir.glob("EDGE_*"))
    ]

    data_list = [
        (data_process_node_for_forward_history(ndf), data_process_edge(edf))
        for ndf, edf in data_list
    ]

    return data_list


batch_size = 64
input_sequence_length = 12
forecast_horizon = 3
multi_horizon = False

if __name__ == "__main__":
    from sys import argv

    datalist = data_read_dir(Path(argv[1]))

    for nodes, edges in datalist:
        print(nodes)
        print(edges)
        for t in create_tf_dataset(
            nodes,
            input_sequence_length=input_sequence_length,
            forecast_horizon=forecast_horizon,
        ):
            print("t")
            print(t)

