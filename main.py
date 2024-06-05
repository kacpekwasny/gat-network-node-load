import random as r
import networkx as nx

from dataclasses import dataclass, field


@dataclass
class DstProbability:
    nodes: list[nx._Node]
    probabilities: list[float]


@dataclass
class NodeAtributes:
    node: int
    routing_table: dict[int, list[int]]

    # list of packets to be forwarded
    # the value(int) represents the destination
    buffer_odd: list[int] = field(default_factory=list)
    buffer_even: list[int] = field(default_factory=list)

    # list of number of packets forwarded in every turn
    forward_history: list[int] = field(default_factory=list)

    def buffers(self, turn: int) -> tuple[list[int], list[int]]:
        if turn % 2 == 0:
            out_buf = self.buffer_even
            next_buf = self.buffer_odd
        else:
            out_buf = self.buffer_odd
            next_buf = self.buffer_even
        return out_buf, next_buf


def build_random_net() -> nx.Graph:
    nodes_no = r.randint(5, 20)
    g: nx.Graph = nx.cycle_graph(nodes_no)

    more_edges = r.randint(0, (nodes_no * (nodes_no) / 2 - nodes_no) // 2)

    while more_edges > 0:
        n1 = r.choice(g.nodes)
        n2 = r.choice(g.nodes)
        if n1 == n2:
            continue
        # if n1 and n2 conneted, then nothing changed
        g.add_edge(n1, n2)
        more_edges -= 1

    return g


def generate_dst_probability_random_uniform(g: nx.Graph) -> DstProbability:
    return DstProbability(
        nodes=list(g.nodes),
        probabilities=[1/len(g.nodes) for n in g.nodes],
    )


def dst_choice(p: DstProbability, k: int = 1) -> nx._Node:
    for i in range(k):
        yield r.choices(population=p.nodes, weights=p.probabilities, k=k)


def simulation(g: nx.Graph, turns: int):
    # network
    # routing tables
    # plan traffic (who generates traffic???)
    # start: while planned traffic
    #   for nodes:
    #       for packets in self.input_buffer:
    #           check next hop in routing table
    #           send to next hop
    #           increment forwarded packet
    nodes: dict[int, NodeAtributes] = {
        n: NodeAtributes(n, nx.shortest_path(g, source=n))
        for n in g.nodes
    }

    turn = turns

    while turn:
        for n, node in nodes.items():
            buf_out, _ = node.buffers()
            node.forward_history.append(len(buf_out))

            while len(buf_out):
                packet = buf_out.pop()
                # the packet is at the destination
                if packet == n:
                    continue
                next_hop = node.routing_table[n][1]
                nodes[next_hop].buffers(turns)
