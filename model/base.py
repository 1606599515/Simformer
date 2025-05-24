import torch
from torch import nn
from torch_scatter import scatter_sum


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


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers,
                       layer_norm=False, activate_final=False)

    def forward(self, node_features):
        return self.mlp(node_features)


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