import torch


def main():
    data = torch.load('data.pt')
    print(f'Data: {data}')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Node feature shape: {data.x.shape}')
    print(f'Edge index shape: {data.edge_index.shape}')
    print(f'Unique labels: {data.y.unique()}')
    print(f'Number of unique labels: {data.y.unique().size(0)}')


if __name__ == "__main__":
    main()
