import json
import random as r
import networkx as nx
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from pathlib import Path

from pandas.compat import sys

DATA_DIR = Path(__file__).parent / "data"

# notes:
# DATA:
#   X:          189, 3,   floaty: (0, 1)
#   Edge Index: 2, 9375,  inty
#   Edge Attr:  9375, 10, floaty (-1, 30)
#   Y:          189, 1    floaty (..., 25)

### zafixowane, bo architektura ###
SMALLEST_NET = 50
BIGGEST_NET = 50

MIN_PACKETS_PER_TURN = 0
MAX_PACKETS_PER_TURN = 20
MORE_PACKETS_PER_TURN = 10


@dataclass
class DstProbability:
    nodes: list[int]
    probabilities: list[float]


@dataclass
class NetworkNode:
    node: int
    routing_table: dict[int, list[int]]

    # list of packets to be forwarded
    # the value(int) represents the destination
    buffer_odd: list[int] = field(default_factory=list)
    buffer_even: list[int] = field(default_factory=list)

    # list of number of packets forwarded in every turn
    forward_history: list[int] = field(default_factory=list)

    dst_probability: DstProbability = field(default=None)

    # probability of generating packets, the min every turn, and max
    packet_generated_probability: tuple[int, int] = field(
            default_factory=lambda: (
                lambda x: [x, x + r.randint(0, MORE_PACKETS_PER_TURN)])(
                r.randint(MIN_PACKETS_PER_TURN, MAX_PACKETS_PER_TURN)
    ))

    generated_packets: int = field(default=0)

    def set_dst_probability(self, probability_generator, g: nx.Graph):
        self.dst_probability = probability_generator(self.node, g)

    def buffers(self, turn: int) -> tuple[list[int], list[int]]:
        if turn % 2 == 0:
            out_buf = self.buffer_even
            next_buf = self.buffer_odd
        else:
            out_buf = self.buffer_odd
            next_buf = self.buffer_even
        return out_buf, next_buf

    def generate_traffic(self, k: int) -> list[int]:
        return list(dst_choice(self.dst_probability, k))


def build_random_graph(smallest: int, biggest: int) -> nx.Graph:
    nodes_no = r.randint(smallest, biggest)
    g: nx.Graph = nx.cycle_graph(nodes_no)

    more_edges = r.randint(0, (nodes_no * (nodes_no - 1) // 2) // 2)

    while more_edges > 0:
        n1 = r.choice(list(g.nodes))
        n2 = r.choice(list(g.nodes))
        if n1 == n2 or g.has_edge(n1, n2):
            continue
        # if n1 and n2 connected, then nothing changed
        g.add_edge(n1, n2)
        more_edges -= 1

    return g


def generate_dst_probability_random_uniform(self: int, g: nx.Graph) -> DstProbability:
    nodes = list(g.nodes)
    nodes.remove(self)
    return DstProbability(
        nodes=nodes,
        probabilities=[1/len(nodes) for _ in nodes],
    )


def dst_choice(p: DstProbability, k: int = 1) -> list[int]:
    return r.choices(population=p.nodes, weights=p.probabilities, k=k)


def simulation(g: nx.Graph, turns: int) -> dict[int, NetworkNode]:
    # network
    # routing tables
    # plan traffic (who generates traffic???)
    # start: while planned traffic
    #   for nodes:
    #       for packets in self.input_buffer:
    #           check next hop in routing table
    #           send to next hop
    #           increment forwarded packet
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
            buf_out, buf_next = node.buffers(turn)
            node.forward_history.append(len(buf_out))

            while buf_out:
                packet = buf_out.pop()
                # the packet is at the destination
                if packet == n:
                    continue
                next_hop = node.routing_table[packet][1]
                _, buf2_next = nodes[next_hop].buffers(turn)
                buf2_next.append(packet)

            k = how_many_new_packets(node)
            node.generated_packets += k
            for new_packet in node.generate_traffic(k):
                buf_next.append(new_packet)

        turn -= 1

    return nodes


def generate_node_and_edge_data(
        smallest_net: int,
        biggest_net: int,
        turns: int,
        to_csv_and_suff: tuple[bool, str] = (False, "1"),
) -> tuple[pd.DataFrame, pd.DataFrame]:

    to_csv, suffix = to_csv_and_suff

    g = build_random_graph(smallest_net, biggest_net)

    nodes = simulation(g, turns)
    if to_csv:
        nx.draw(g)
        plt.savefig(DATA_DIR / f"PLOT_{suffix}.png")

    # Save node data
    node_data = []
    for n, node in nodes.items():
        routing = [len(r) for r in node.routing_table.values()]
        node_data.append({
            'node': n,
            'routing': json.dumps((np.array(routing) / max(routing)).tolist()),
            'forward_min': min(node.forward_history),
            'forward_max': max(node.forward_history),
            'forward_avg': sum(node.forward_history) / turns,
            'generated_packets_sum': node.generated_packets,
            'generated_packets_avg': node.generated_packets / turns,
            'generated_packets_prob_min': node.packet_generated_probability[0],
            'generated_packets_prob_max': node.packet_generated_probability[1],
        })
    node_df = pd.DataFrame(node_data)
    if to_csv:
        node_df.to_csv(DATA_DIR / f'NODE_{suffix}.csv', index=False)

    # Save graph structure
    edge_idx = np.array(list(g.edges), dtype=np.longlong)
    edge_idx = np.concatenate((edge_idx, edge_idx[:, [1, 0]]))
    edge_df = pd.DataFrame(edge_idx, columns=['source', 'target'])
    if to_csv:
        edge_df.to_csv(DATA_DIR / f'EDGE_{suffix}.csv', index=False)

    return node_df, edge_df


if __name__ == "__main__":
    n = int(sys.argv[1])
    for i in range(n):
        print("Generating ", i, "/", n)
        node_data, edge_data = generate_node_and_edge_data(
                50, 50, 5000, to_csv_and_suff=(True, str(i)))
