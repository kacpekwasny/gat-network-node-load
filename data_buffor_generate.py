import json
import random as r
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pathlib import Path
from os import makedirs


'''
    Class to store the probability of sending a packet to a given node.
'''
@dataclass
class DstProbability:
    nodes: list[int]
    probabilities: list[float]


'''
    Class to store the information about a network node.
'''
@dataclass
class NetworkNode:
    node: int
    routing_table: dict[int, list[int]]

    buffer: list[int] = field(default_factory=list)

    forward_history: list[int] = field(default_factory=list)

    dst_probability: DstProbability = field(default=None)

    packet_generated_probability: tuple[int, int] = field(
        default_factory=lambda: (
            lambda x: [x, x + r.randint(0, MORE_PACKETS_PER_TURN)])(
            r.randint(MIN_PACKETS_PER_TURN, MAX_PACKETS_PER_TURN)
        ))
    generated_packets: int = field(default=0)

    def __post_init__(self):
        return

    def set_dst_probability(self, probability_generator, g: nx.Graph):
        self.dst_probability = probability_generator(self.node, g)

    def generate_traffic(self, k: int) -> list[int]:
        return list(dst_choice(self.dst_probability, k))


'''
    Generate a random graph with a random number of nodes and edges.
    The number of nodes is between smallest and biggest.
'''
def build_random_graph(smallest: int, biggest: int) -> nx.Graph:
    nodes_no = r.randint(smallest, biggest)
    g: nx.Graph = nx.cycle_graph(nodes_no)
    more_edges = r.randint(0, (nodes_no * (nodes_no - 1) // 2) // 2)
    while more_edges > 0:
        n1 = r.choice(list(g.nodes))
        n2 = r.choice(list(g.nodes))
        if n1 == n2 or g.has_edge(n1, n2):
            continue
        g.add_edge(n1, n2)
        more_edges -= 1
    return g


'''
    Generate the same probability of sending a packet to each node.
'''
def generate_dst_probability_random_uniform(self: int, g: nx.Graph) -> DstProbability:
    nodes = list(g.nodes)
    nodes.remove(self)
    return DstProbability(
        nodes=nodes,
        probabilities=[1/len(nodes) for _ in nodes],
    )


'''
    Choosing the destination node based on the probability.
'''
def dst_choice(p: DstProbability, k: int = 1) -> list[int]:
    return r.choices(population=p.nodes, weights=p.probabilities, k=k)


'''
    Simulate the network for a given number of turns.
'''
def simulation(g: nx.Graph, turns: int, packets_per_tick: int) -> dict[int, NetworkNode]:
    nodes: dict[int, NetworkNode] = {
        n: NetworkNode(n, nx.shortest_path(g, source=n))
        for n in g.nodes
    }

    def how_many_new_packets(n: NetworkNode) -> int:
        return r.randint(*n.packet_generated_probability)

    for node in nodes.values():
        node.set_dst_probability(generate_dst_probability_random_uniform, g)

    turn = turns
    while turn:
        for n, node in nodes.items():
            node.forward_history.append(len(node.buffer))

            # Send packets from buffer
            for _ in range(min(packets_per_tick, len(node.buffer))):
                packet = node.buffer.pop(0)
                if packet == n:
                    continue
                next_hop = node.routing_table[packet][1]
                nodes[next_hop].buffer.append(packet)

            # Generate new packets
            k = how_many_new_packets(node)
            node.generated_packets += k
            for new_packet in node.generate_traffic(k):
                node.buffer.append(new_packet)

        turn -= 1

    return nodes


'''
    Generate the data for the nodes and edges based on the simulation.
'''
def generate_node_and_edge_data(
        smallest_net: int,
        biggest_net: int,
        turns: int,
        packets_per_tick: int,
        to_csv_and_suff: tuple[bool, str] = (False, "1"),
) -> tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:

    to_csv, suffix = to_csv_and_suff

    g = build_random_graph(smallest_net, biggest_net)
    nodes = simulation(g, turns, packets_per_tick)
    if to_csv:
        nx.draw(g)
        plt.savefig(DATA_DIR / f"PLOT_{suffix}.png")

    node_data = []
    for n, node in nodes.items():
        no_of_hops = [len(node.routing_table[r]) for r in sorted(node.routing_table.keys())]
        node_data.append({
            'node': n,
            'routing': json.dumps(no_of_hops),
            'forward_history': json.dumps(node.forward_history),
            'forward_min': min(node.forward_history),
            'forward_max': max(node.forward_history),
            'forward_avg': sum(node.forward_history) / len(node.forward_history),
            'generated_packets_sum': node.generated_packets,
            'generated_packets_avg': node.generated_packets / turns,
            'generated_packets_prob_min': node.packet_generated_probability[0],
            'generated_packets_prob_max': node.packet_generated_probability[1],
            'buffer_size': max(len(node.buffer), len(node.buffer))
        })
    node_df = pd.DataFrame(node_data)

    edge_idx = np.array(list(g.edges), dtype=np.longlong)
    edge_idx = np.concatenate((edge_idx, edge_idx[:, [1, 0]]))
    edge_df = pd.DataFrame(edge_idx, columns=['source', 'target'])

    return g, node_df, edge_df


'''
    Save the network simulation data to the CSV files.
'''
def save_network_sim_data(edge_df: pd.DataFrame, node_df: pd.DataFrame, suffix: str, data_dir=None):
    if data_dir is None:
        data_dir = DATA_DIR
        
    edge_df.to_csv(data_dir / f'EDGE_{suffix}.csv', index=False)
    node_df.to_csv(data_dir / f'NODE_{suffix}.csv', index=False)

DATA_DIR = Path(__file__).parent / "data_size_5"
SMALLEST_NET = 50
BIGGEST_NET = 50
MIN_PACKETS_PER_TURN = 2
MAX_PACKETS_PER_TURN = 4
MORE_PACKETS_PER_TURN = 1
SIM_TICKS = 5000

if __name__ == "__main__":
    from sys import argv
    if len(argv) > 2:
        DATA_DIR = DATA_DIR.parent / argv[2]
    makedirs(DATA_DIR, exist_ok=True)

    n = int(argv[1])
    packets_per_tick = 6  #! Limit the number of packets processed per tick
    for i in range(n):
        print("Generating ", i, "/", n)
        g, node_data, edge_data = generate_node_and_edge_data(SMALLEST_NET, BIGGEST_NET, SIM_TICKS, packets_per_tick)

        if True:
            save_network_sim_data(edge_data, node_data, suffix=str(i), data_dir=DATA_DIR)
