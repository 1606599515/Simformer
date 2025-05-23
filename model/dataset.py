from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering

import networkx as nx

from sklearn.neighbors import NearestNeighbors


def collate_fn(batch):
    # Find the maximum number of nodes (N) and edges (E) in the batch
    max_nodes = max(data['pos'].shape[0] + 1 for data, _, _ in batch)
    max_edges = max(data['connections'].shape[0] for data, _, _ in batch)

    # Initialize lists to hold the padded data
    padded_pos = []
    padded_node = []
    padded_connections = []
    padded_mask = []
    padded_output = []

    for data, idx, path in batch:
        pos = data['pos']
        node = data['node']
        connections = data['connections']
        output = data['output']

        current_node_num = pos.shape[0]

        # Pad nodes
        pad_size = max_nodes - current_node_num
        pos = torch.cat([pos, torch.zeros((pad_size, pos.shape[1]))], dim=0)
        node = torch.cat([node, torch.zeros((pad_size, node.shape[1]))], dim=0)
        output = torch.cat([output, torch.zeros((pad_size, output.shape[1]))], dim=0)

        # Pad edges
        edge_pad_size = max_edges - connections.shape[0]
        connections = torch.cat([connections, torch.full((edge_pad_size, connections.shape[1]), max_nodes-1)], dim=0)

        # Create mask for nodes
        mask = torch.cat([torch.ones(current_node_num), torch.zeros(pad_size)], dim=0)

        padded_pos.append(pos)
        padded_node.append(node)
        padded_connections.append(connections)
        padded_output.append(output)
        padded_mask.append(mask)

    # Stack all padded data
    batch_pos = torch.stack(padded_pos)
    batch_node = torch.stack(padded_node)
    batch_connections = torch.stack(padded_connections)
    batch_output = torch.stack(padded_output)
    batch_mask = torch.stack(padded_mask)

    return dict(pos=batch_pos,
                node=batch_node,
                output=batch_output,
                connections=batch_connections,
                mask=batch_mask), [idx for _, idx, _ in batch], [path for _, _, path in batch]


def cluster_collate_fn(batch):
    # Find the maximum number of nodes (N) and edges (E) in the batch
    max_nodes = max(data['pos'].shape[0] + 1 for data, _, _ in batch)
    max_edges = max(data['connections'].shape[0] for data, _, _ in batch)

    # Initialize lists to hold the padded data
    padded_pos = []
    padded_node = []
    padded_connections = []
    padded_mask = []
    padded_output = []
    padded_cluster = []

    for data, idx, path in batch:
        pos = data['pos']
        node = data['node']
        connections = data['connections']
        output = data['output']
        cluster_matrix = data['cluster_matrix']

        current_node_num = pos.shape[0]

        # Pad nodes
        pad_size = max_nodes - current_node_num
        pos = torch.cat([pos, torch.zeros((pad_size, pos.shape[1]))], dim=0)
        node = torch.cat([node, torch.zeros((pad_size, node.shape[1]))], dim=0)
        output = torch.cat([output, torch.zeros((pad_size, output.shape[1]))], dim=0)
        cluster_matrix = torch.cat([cluster_matrix, torch.zeros((pad_size, cluster_matrix.shape[1]))], dim=0)

        # Pad edges
        edge_pad_size = max_edges - connections.shape[0]
        connections = torch.cat([connections, torch.full((edge_pad_size, connections.shape[1]), max_nodes-1)], dim=0)

        # Create mask for nodes
        mask = torch.cat([torch.ones(current_node_num), torch.zeros(pad_size)], dim=0)

        padded_pos.append(pos)
        padded_node.append(node)
        padded_connections.append(connections)
        padded_output.append(output)
        padded_mask.append(mask)
        padded_cluster.append(cluster_matrix)

    # Stack all padded data
    batch_pos = torch.stack(padded_pos)
    batch_node = torch.stack(padded_node)
    batch_connections = torch.stack(padded_connections)
    batch_output = torch.stack(padded_output)
    batch_mask = torch.stack(padded_mask)
    batch_cluster = torch.stack(padded_cluster)

    return dict(pos=batch_pos,
                node=batch_node,
                output=batch_output,
                connections=batch_connections,
                mask=batch_mask,
                cluster_matrix=batch_cluster), [idx for _, idx, _ in batch], [path for _, _, path in batch]


def get_beam_data(path):

    data = np.load(os.path.join(path), mmap_mode='r')

    pos = data['positions']
    connections = data['connections']
    output = data['mises']

    border = data['border']
    constraint = data['constraint']
    load = data['load']

    node = np.concatenate((border, constraint, load), axis=1)

    return pos, node, connections, output


class BeamDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        super(BeamDataset, self).__init__()

        assert data_path is not None, f"Data path {data_path} is None"
        assert mode in ['train', 'test', 'valid'], "Mode should be either 'train', 'test', or 'valid'"

        self.data_path = data_path
        self.mode = mode

        self.data = np.load(os.path.join(self.data_path, '%s.npy' % mode), allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, '%d.npz' % self.data[idx])
        pos, node, connections, output = get_beam_data(path)
        # pos [N, 2]
        # node [N, 4]
        # connections [E, 2]
        # output [N, 1]

        # pos [N, 2]
        pos = torch.from_numpy(pos).float()

        # node [N, 4]
        node = torch.from_numpy(node).float()

        # connections [E, 2]
        connections = torch.from_numpy(connections).long()

        # output [N, 1]
        output = torch.from_numpy(output).float()

        return dict(pos=pos,
                    node=node,
                    connections=connections,
                    output=output), idx, self.data[idx]


def kmeans(pos, num_clusters):

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
    labels = kmeans.fit_predict(pos)

    N = pos.shape[0]
    relation_matrix = np.zeros((N, num_clusters), dtype=np.float32)
    relation_matrix[np.arange(N), labels] = 1

    return relation_matrix


def spectral_clustering(pos, num_clusters):

    spectral = SpectralClustering(n_clusters=num_clusters, affinity='nearest_neighbors', random_state=0)
    labels = spectral.fit_predict(pos)

    N = pos.shape[0]
    relation_matrix = np.zeros((N, num_clusters), dtype=np.float32)
    relation_matrix[np.arange(N), labels] = 1

    return relation_matrix


def networkx_clustering(pos, connections, num_clusters):

    N = pos.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(connections)

    if not nx.is_connected(G):

        connected_components = list(nx.connected_components(G))
        relation_matrix = np.zeros((N, num_clusters), dtype=np.float32)
        cluster_id = 0

        for component in connected_components:
            subgraph = G.subgraph(component)
            subgraph_nodes = list(subgraph.nodes)

            if len(subgraph_nodes) < num_clusters:
                for node in subgraph_nodes:
                    relation_matrix[node, cluster_id % num_clusters] = 1
                    cluster_id += 1
            else:
                sub_communities = nx.algorithms.community.asyn_fluidc(subgraph, k=min(num_clusters, len(subgraph_nodes)))
                for sub_cluster_id, community in enumerate(sub_communities):
                    for node in community:
                        relation_matrix[node, sub_cluster_id] = 1
    else:

        communities = nx.algorithms.community.asyn_fluidc(G, k=num_clusters)
        relation_matrix = np.zeros((N, num_clusters), dtype=np.float32)
        for cluster_id, community in enumerate(communities):
            for node in community:
                relation_matrix[node, cluster_id] = 1

    return relation_matrix


class ClusterDataset(Dataset):
    def __init__(self, data_path, mode='train', config=None):
        super(ClusterDataset, self).__init__()

        assert data_path is not None, f"Data path {data_path} is None"
        assert mode in ['train', 'test', 'valid'], "Mode should be either 'train', 'test', or 'valid'"

        self.data_path = data_path
        self.mode = mode

        self.data = np.load(os.path.join(self.data_path, '%s.npy' % mode), allow_pickle=True)

        if config is not None:
            self.config = config

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, '%d.npz' % self.data[idx])
        pos, node, connections, output = get_beam_data(path)
        # pos [N, 2]
        # node [N, 4]
        # connections [E, 2]
        # output [N, 1]

        pos = torch.from_numpy(pos).float()

        node = torch.from_numpy(node).float()

        # connections [E, 2]
        connections = torch.from_numpy(connections).long()

        # output [N, 1]
        output = torch.from_numpy(output).float()

        cluster_path = os.path.join(self.data_path, self.config.clustering_methods, str(self.config.num_clusters))
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)

        if self.config.clustering_methods == 'kmeans':
            if os.path.exists(os.path.join(cluster_path, '%d.npz' % self.data[idx])):
                cluster_matrix = np.load(os.path.join(cluster_path, '%d.npz' % self.data[idx]))['cluster_matrix']
            else:
                cluster_matrix = kmeans(pos, self.config.num_clusters)

                np.savez(os.path.join(cluster_path, '%d.npz' % self.data[idx]), cluster_matrix=cluster_matrix)
        elif self.config.clustering_methods == 'spectral_clustering':
            if os.path.exists(os.path.join(cluster_path, '%d.npz' % self.data[idx])):
                cluster_matrix = np.load(os.path.join(cluster_path, '%d.npz' % self.data[idx]))['cluster_matrix']
            else:
                cluster_matrix = spectral_clustering(pos, self.config.num_clusters)

                np.savez(os.path.join(cluster_path, '%d.npz' % self.data[idx]), cluster_matrix=cluster_matrix)
        elif self.config.clustering_methods == 'metis':
            if os.path.exists(os.path.join(cluster_path, '%d.npz' % self.data[idx])):
                cluster_matrix = np.load(os.path.join(cluster_path, '%d.npz' % self.data[idx]))['cluster_matrix']
            else:
                cluster_matrix = networkx_clustering(pos, connections, self.config.num_clusters)

                np.savez(os.path.join(cluster_path, '%d.npz' % self.data[idx]), cluster_matrix=cluster_matrix)
        else:

            cluster_matrix = np.ones([pos.shape[0], self.config.num_clusters], dtype=np.float32)
        cluster_matrix = torch.from_numpy(cluster_matrix).float()

        return dict(pos=pos,
                    node=node,
                    connections=connections,
                    cluster_matrix=cluster_matrix,
                    output=output), idx, self.data[idx]


def get_sw_data(path):

    data = np.load(os.path.join(path), mmap_mode='r')

    pos = data['positions']
    connections = data['connections']
    output = data['stress']

    constraint = data['constraint']
    load = data['load']
    load = np.tile(load, (1, 3)) * np.tile(np.array([[0.0, -700.0, 0.0]]), (pos.shape[0], 1)) / load.sum()

    node = np.concatenate((constraint, load), axis=1)

    return pos, node, connections, output


class SWDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        super(SWDataset, self).__init__()

        assert data_path is not None, f"Data path {data_path} is None"
        assert mode in ['train', 'test', 'valid'], "Mode should be either 'train', 'test', or 'valid'"

        self.data_path = data_path
        self.mode = mode

        self.data = np.load(os.path.join(self.data_path, '%s.npy' % mode), allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, '%d.npz' % self.data[idx])
        pos, node, connections, output = get_sw_data(path)

        pos = torch.from_numpy(pos).float()

        node = torch.from_numpy(node).float()

        connections = torch.from_numpy(connections).long()

        output = torch.from_numpy(output).float()

        return dict(pos=pos,
                    node=node,
                    connections=connections,
                    output=output), idx, self.data[idx]


def get_elasticity_data(path):

    data = np.load(os.path.join(path), mmap_mode='r')

    pos = data['positions']
    output = data['stress']
    connections = data['connections']

    node = pos

    return pos, node, connections, output


class ElasticityDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        super(ElasticityDataset, self).__init__()

        assert data_path is not None, f"Data path {data_path} is None"
        assert mode in ['train', 'test', 'valid'], "Mode should be either 'train', 'test', or 'valid'"

        self.data_path = data_path
        self.mode = mode
        if self.mode == 'train':
            self.data = np.array(range(1600))
        elif self.mode == 'test':
            self.data = np.array(range(1600, 1800))
        elif self.mode == 'valid':
            self.data = np.array(range(1800, 2000))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, '%d.npz' % self.data[idx])

        pos, node, connections, output = get_elasticity_data(path)

        pos = torch.from_numpy(pos).float()

        node = torch.from_numpy(node).float()

        connections = torch.from_numpy(connections).long()

        output = torch.from_numpy(output).float()

        return dict(pos=pos,
                    node=node,
                    connections=connections,
                    output=output), idx, self.data[idx]


def get_car_data(path):

    from dgl.data.utils import load_graphs
    graphs, labels = load_graphs(path)
    graph = graphs[0]
    pos = graph.ndata['positions'].numpy()
    output = graph.ndata['gt'].reshape(-1, 1).numpy()
    node = pos

    return pos, node, output


class DrivAerNetDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        super(DrivAerNetDataset, self).__init__()

        assert data_path is not None, f"Data path {data_path} is None"
        assert mode in ['train', 'test', 'valid'], "Mode should be either 'train', 'test', or 'valid'"

        self.data_path = data_path
        self.mode = mode

        self.data = np.load(os.path.join(self.data_path, '%s.npy' % mode), allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = os.path.join(self.data_path, f'{self.data[idx]}.bin')
        pos, node, output = get_car_data(path)

        if not os.path.exists(os.path.join(self.data_path, 'connection')):
            os.makedirs(os.path.join(self.data_path, 'connection'))

        connection_path = os.path.join(self.data_path, 'connection', f'{self.data[idx]}.npy')
        if os.path.exists(connection_path):
            connections = np.load(connection_path)
        else:
            k = 6
            knn = NearestNeighbors(n_neighbors=k + 1)
            knn.fit(pos)
            distances, receivers = knn.kneighbors(pos)

            receivers = receivers[:, 1:]

            senders = np.repeat(np.arange(pos.shape[0]), k)
            receivers = receivers.reshape(-1)

            connections = np.stack([senders, receivers], axis=1)
            connections = np.vstack((connections, connections[:, [1, 0]]))
            connections = np.unique(connections, axis=0)

            np.save(connection_path, connections)

        pos = torch.from_numpy(pos).float()

        node = torch.from_numpy(node).float()

        connections = torch.from_numpy(connections).long()

        output = torch.from_numpy(output).float()

        return dict(pos=pos,
                    node=node,
                    connections=connections,
                    output=output), idx, self.data[idx]


if __name__ == '__main__':

    data_path = os.path.join('E:\\', 'Project', 'Elasticity', 'data')
    dataset = ElasticityDataset(data_path=data_path, mode='train')

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_edges = 0
    num_samples = len(dataset)

    for data, _, _ in data_loader:
        connections = data['connections']  # [E, 2]
        total_edges += connections.shape[1]

    average_edges = total_edges / num_samples
    print(f"average edges: {average_edges:.2f}")

