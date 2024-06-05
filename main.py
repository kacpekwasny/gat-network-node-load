import random as r
import networkx as nx

from dataclasses import dataclass


@dataclass
class DstProbability:
    nodes: list[nx._Node]
    probabilities: list[float]


@dataclass
class NodeAtributes:
    node: int
    # list of number of packets forwarded in every turn
    forward_history: list[int]
    routing_table: dict[int, list[int]]


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


def simulation(g: nx.Graph):
    # network
    # routing tables
    # plan traffic (who generates traffic???)
    # start: while planned traffic
    #   for nodes:
    #       for packets in self.input_buffer:
    #           check next hop in routing table
    #           send to next hop
    #           increment forwarded packet
    nodes = {n: NodeAtributes(n, 0, nx.shortest_path(g, source=n))
             for n in g.nodes}
    
