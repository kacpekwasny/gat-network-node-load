import random as r
import networkx as nx
import pandas as pd
from dataclasses import dataclass, field


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

    def generate_traffic_for_turn(self, k: int) -> list[int]:
        return list(dst_choice(self.dst_probability, k))


def build_random_graph() -> nx.Graph:
    nodes_no = r.randint(350, 450)
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


def dst_choice(p: DstProbability, k: int = 1) -> int:
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

    for node in nodes.values():
        node.set_dst_probability(generate_dst_probability_random_uniform, g)

    def random_number_of_packets(n: int) -> int:
        return r.randint(0, 20)

    how_many_new_packets = random_number_of_packets
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

            for new_packet in node.generate_traffic_for_turn(how_many_new_packets(n)):
                buf_next.append(new_packet)

        turn -= 1

    return nodes

def main():
    g = build_random_graph()
    nodes = simulation(g, 50)

    # Save node data
    node_data = []
    for n, node in nodes.items():
        node_data.append({
            'node': n,
            'forward_history': node.forward_history
        })
    df = pd.DataFrame(node_data)
    df.to_csv('network_data.csv', index=False)

    # Save graph structure
    edges = list(g.edges)
    edge_df = pd.DataFrame(edges, columns=['source', 'target'])
    edge_df.to_csv('network_edges.csv', index=False)

if __name__ == "__main__":
    main()
