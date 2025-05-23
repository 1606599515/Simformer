import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean
import random
import numpy as np
from model.cluster_transformer import ClusterTransformer
from model.normalizer import Normalizer
from model.seed import set_seed


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 layer_norm=True,
                 activation=nn.ReLU(),
                 activate_final=False):

        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation]
        for i in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation]

        if activate_final:
            layers += [nn.Linear(hidden_dim, output_dim), activation]
        else:
            layers += [nn.Linear(hidden_dim, output_dim)]

        if layer_norm:
            layers += [nn.LayerNorm(output_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, input):
        output = self.net(input)

        return output


class GraphNetBlock(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(GraphNetBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.mlp_node = MLP(input_dim=2 * hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers)  # 3*hidden_dim: [nodes, accumulated_edges]
        self.mlp_edge = MLP(input_dim=3 * hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers)  # 3*hidden_dim: [sender, edge, receiver]

    def update_edges(self, senders, receivers, edge_features, sender_node_features, receiver_node_features):
        # print("Senders min/max:", senders.min().item(), senders.max().item())
        # print("Receivers min/max:", receivers.min().item(), receivers.max().item())

        senders = senders.unsqueeze(2).expand(-1, -1, self.hidden_dim)
        receivers = receivers.unsqueeze(2).expand(-1, -1, self.hidden_dim)

        if senders.max().item() >= sender_node_features.shape[1] or receivers.max().item() >= sender_node_features.shape[1]:
            raise ValueError("Index out of bound in senders or receivers")

        sender_features = torch.gather(sender_node_features, 1, senders)
        receiver_features = torch.gather(receiver_node_features, 1, receivers)

        features = torch.cat([sender_features, receiver_features, edge_features], dim=-1)
        return self.mlp_edge(features)

    def update_nodes(self, receivers, edge_features, receiver_node_features):
        accumulate_edges = scatter_sum(edge_features, receivers, dim=1)  # ~ tf.math.unsorted_segment_sum
        if receiver_node_features.shape[1] != accumulate_edges.shape[1]:
            if receiver_node_features.shape[1] - torch.max(receivers) - 1 > 0:
                zeros_row = torch.zeros((receiver_node_features.shape[0],
                                         receiver_node_features.shape[1] - torch.max(receivers) - 1,
                                         accumulate_edges.shape[-1]))
                accumulate_edges = torch.cat([accumulate_edges, zeros_row.cuda()], dim=1)
        features = torch.cat([receiver_node_features, accumulate_edges], dim=-1)

        return self.mlp_node(features)

    def forward(self, senders, receivers, edge_features, sender_node_features, receiver_node_features):
        new_edge_features = self.update_edges(senders, receivers, edge_features, sender_node_features,
                                              receiver_node_features)
        new_receiver_node_features = self.update_nodes(receivers, new_edge_features, receiver_node_features)
        new_receiver_node_features += receiver_node_features
        new_edge_features += edge_features

        return new_edge_features, new_receiver_node_features


class Encoder(nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.node_mlp = MLP(input_dim=input_dim_node, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers, activate_final=False)

        self.edge_mlp = MLP(input_dim=input_dim_edge, hidden_dim=hidden_dim, output_dim=hidden_dim,
                            num_layers=num_layers, activate_final=False)

    def forward(self, node_features, edge_features):
        edge_latents = self.edge_mlp(edge_features)
        node_latents = self.node_mlp(node_features)

        return node_latents, edge_latents


class Process(nn.Module):
    def __init__(self, hidden_dim, num_layers, message_passing_steps):
        super(Process, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(message_passing_steps):
            self.blocks.append(GraphNetBlock(hidden_dim, num_layers))

    def forward(self, senders, receivers, edge_features, sender_node_features, receiver_node_features, sr=True):
        for graphnetblock in self.blocks:
            edge_features, receiver_node_features = graphnetblock(senders, receivers, edge_features,
                                                                  sender_node_features,
                                                                  receiver_node_features)
            if sr:
                sender_node_features = receiver_node_features

        return edge_features, receiver_node_features


class TransformerProcess(nn.Module):
    def __init__(self, hidden_dim, num_layers, steps, num_clusters):
        super(TransformerProcess, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(steps):
            self.blocks.append(ClusterTransformer(dim=hidden_dim, num_clusters=num_clusters))

    def forward(self, node_features, mask):
        node_cluster_attention = None
        all_clustering_weights = []
        for transformerblock in self.blocks:
            node_features, node_cluster_attention, clustering_weights = transformerblock(node_features, node_cluster_attention, mask)
            all_clustering_weights.append(clustering_weights)

        all_clustering_weights = torch.stack(all_clustering_weights)
        return node_features, all_clustering_weights


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       layer_norm=False, activate_final=False)

    def forward(self, node_features):
        return self.mlp(node_features)


def generate_adjacency_matrix(connections, num_nodes, mask):
    """
    Generate an adjacency matrix based on connections and mask.
    :param connections: [B, E, 2], each row represents (sender, receiver)
    :param num_nodes: Total number of nodes
    :param mask: [B, N], validity mask for nodes
    :return: Adjacency matrix [B, N, N]
    """
    B, E, _ = connections.shape
    adj = torch.zeros(B, num_nodes, num_nodes, device=connections.device)  # Initialize adjacency matrix to 0

    # Extract sender and receiver indices
    senders, receivers = connections[..., 0], connections[..., 1]  # [B, E]

    # Generate batch indices
    batch_indices = torch.arange(B, device=connections.device).view(-1, 1).expand(-1, E)  # [B, E]

    # Fill adjacency matrix
    adj[batch_indices, senders, receivers] = 1  # Fill 1 at receiver positions
    adj[batch_indices, receivers, senders] = 1  # Fill 1 at sender positions (undirected graph)

    # Use mask to set rows and columns of invalid nodes to 0
    mask = mask.unsqueeze(-1)  # [B, N, 1]
    adj = adj * mask  # Set rows of invalid nodes to 0
    adj = adj * mask.transpose(1, 2)  # Set columns of invalid nodes to 0

    return adj


def compute_loss(adj, all_clustering_weights):
    """
    Compute the loss based on adj and all_clustering_weights (without explicitly using K).
    :param adj: [B, N, N], adjacency matrix
    :param all_clustering_weights: [K, B, N, M], clustering weights
    :return: loss
    """
    # Adjust the dimensions of all_clustering_weights to [B, K, N, M]
    all_clustering_weights = all_clustering_weights.permute(1, 0, 2, 3)  # [B, K, N, M]

    # Compute the weighted adjacency matrix
    clustering_weights_T = all_clustering_weights.transpose(-2, -1)  # [B, K, M, N]
    weighted_adj = torch.matmul(clustering_weights_T, torch.matmul(adj.unsqueeze(1), all_clustering_weights))  # [B, K, M, M]

    # Compute the loss, e.g., Frobenius norm
    identity = torch.eye(weighted_adj.size(-1), device=adj.device).unsqueeze(0).unsqueeze(0)  # [1, 1, M, M]
    loss = torch.norm(weighted_adj - identity, p='fro', dim=(-2, -1))  # [B, K]

    # Take the mean over the K and B dimensions
    total_loss = loss.mean()

    return total_loss


def entropy_regularization(all_clustering_weights):
    """
    Compute the entropy regularization loss for all_clustering_weights.
    :param all_clustering_weights: [K, B, N, M], clustering weights
    :return: Entropy regularization loss
    """
    # Take the mean over the K dimension, resulting in shape [B, N, M]
    mean_clustering_weights = all_clustering_weights.mean(dim=0)  # [B, N, M]

    # Avoid log(0)
    mean_clustering_weights = torch.clamp(mean_clustering_weights, min=1e-9)

    # Compute the entropy regularization loss
    entropy_loss = -(mean_clustering_weights * torch.log(mean_clustering_weights)).sum(dim=-1).mean()  # Scalar

    return entropy_loss


class Ours(nn.Module):

    def __init__(self, config):
        super(Ours, self).__init__()

        self.config = config

        self.node_normalizer = Normalizer(size=config.input_dim_node)
        self.edge_normalizer = Normalizer(size=config.input_dim_edge)
        self.output_normalizer = Normalizer(size=config.output_dim)

        self.message_passing_steps = config.message_passing_steps

        self.encoder = Encoder(config.input_dim_node, config.input_dim_edge, config.hidden_dim, config.num_layers)
        self.graph_process = Process(config.hidden_dim, config.num_layers, self.message_passing_steps)
        self.transformer_process = TransformerProcess(config.hidden_dim, config.num_layers, config.transformer_block, config.num_clusters)
        self.decoder = Decoder(config.hidden_dim, config.output_dim, config.num_layers)

    def accumulate(self, node, pos, connections, output):
        self.node_normalizer.accumulate(torch.cat([node, pos], dim=-1))
        senders, receivers = connections[..., 0], connections[..., 1]  # [B, E]

        edge_displacement = pos.gather(1, receivers.unsqueeze(-1).expand(-1, -1, pos.size(-1))) - \
                            pos.gather(1, senders.unsqueeze(-1).expand(-1, -1, pos.size(-1)))  # [B, E, 2]
        edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)  # [B, E, 1]


        edge = torch.cat([edge_displacement, edge_distance], dim=-1)  # [B, E, 3]
        self.edge_normalizer.accumulate(edge)
        self.output_normalizer.accumulate(output)

    def output_normalize(self, data):
        return self.output_normalizer(data)

    def output_normalize_inverse(self, data):
        return self.output_normalizer.inverse(data)

    def forward(self, pos, node, connections, output, mask, noise, mode):

        if noise:

            new_seed = torch.seed()
            if torch.cuda.is_available():
                torch.cuda.manual_seed(new_seed)
                torch.cuda.manual_seed_all(new_seed)

            noise = self.config.noise * torch.randn_like(node)
            node += noise
            set_seed(self.config.seed)

        senders, receivers = connections[..., 0], connections[..., 1]  # [B, E]

        edge_displacement = pos.gather(1, receivers.unsqueeze(-1).expand(-1, -1, pos.size(-1))) - \
                            pos.gather(1, senders.unsqueeze(-1).expand(-1, -1, pos.size(-1)))  # [B, E, 2]
        edge_distance = torch.norm(edge_displacement, dim=-1, keepdim=True)  # [B, E, 1]

        edge = torch.cat([edge_displacement, edge_distance], dim=-1)  # [B, E, 3]

        node_features = self.node_normalizer(torch.cat([node, pos], dim=-1))
        edge_features = self.edge_normalizer(edge)

        node_features, edge_features = self.encoder(node_features, edge_features)

        node_features, all_clustering_weights = self.transformer_process(node_features, mask)

        normalized_output_hat = self.decoder(node_features)

        normalized_output = self.output_normalizer(output)
        output_hat = self.output_normalize_inverse(normalized_output_hat)

        return output_hat, normalized_output, normalized_output_hat



