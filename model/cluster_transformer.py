import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Define linear layers
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        '''
        :param x: [B, N, D], B is batch size, N is sequence length, D is embedding dimension
        :return: updated_x: [B, N, D], attn_matrix: [B, num_heads, N, N]
        '''
        B, N, D = x.shape

        # Linear transformation and split into heads
        Q = self.query_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]
        K = self.key_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)    # [B, num_heads, N, head_dim]
        V = self.value_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, N, N]
        attn_matrix = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N, N]

        # Weighted sum
        attn_output = torch.matmul(attn_matrix, V)  # [B, num_heads, N, head_dim]

        # Concatenate multi-head results
        attn_output = attn_output.transpose(1, 2).reshape(B, N, D)  # [B, N, D]

        # Output linear transformation
        updated_x = self.out_proj(attn_output)  # [B, N, D]

        attn_matrix = attn_matrix.mean(dim=1)  # Average over num_heads dimension, resulting in [B, N, N]

        return updated_x, attn_matrix


class MultiheadCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Define linear layers
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query, key_value, cluster_matrix):
        '''
        :param query: [B, N_q, D_q], B is batch size, N_q is query sequence length, D_q is query feature dimension
        :param key_value: [B, N_kv, D_kv], N_kv is key/value sequence length, D_kv is key/value feature dimension
        :param cluster_matrix: [B, N_q, N_kv], clustering matrix
        :return: updated_query: [B, N_q, D_q], attn_matrix: [B, N_q, N_kv]
        '''
        B, N_q, D_q = query.shape
        _, N_kv, D_kv = key_value.shape

        # Linear transformation and split into heads
        Q = self.query_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_q, head_dim]
        K = self.key_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_kv, head_dim]
        V = self.value_proj(key_value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)  # [B, num_heads, N_kv, head_dim]

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, num_heads, N_q, N_kv]

        # Add cluster_matrix to attn_scores
        attn_scores += cluster_matrix.unsqueeze(1)  # [B, num_heads, N_q, N_kv]

        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, num_heads, N_q, N_kv]

        # Weighted sum
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, N_q, head_dim]

        # Concatenate multi-head results
        attn_output = attn_output.transpose(1, 2).reshape(B, N_q, -1)  # [B, N_q, dim]

        # Output linear transformation
        updated_query = self.out_proj(attn_output)  # [B, N_q, D_q]

        # Average multi-head attention matrix, resulting in [B, N_q, N_kv]
        attn_matrix = attn_weights.mean(dim=1)

        return updated_query, attn_matrix


class AttentionAwareClustering(nn.Module):
    def __init__(self, dim, num_clusters):
        super().__init__()
        self.linear = nn.Linear(dim, num_clusters)
        self.cluster_embeddings = nn.Parameter(torch.randn(num_clusters, dim))

    def forward(self, node_emb, node_cluster_attention, mask):
        '''
        :param node_emb: [B, N, D]
        :param node_cluster_attention: [B, N, C]
        :param mask: [B, N], 1 for real nodes, 0 for padded nodes
        :return:
        '''
        # Compute attention scores
        attn_logits = self.linear(node_emb)  # [B, N, C]
        clustering_weights = attn_logits + node_cluster_attention  # [B, N, C]

        # Apply mask, set weights of padded nodes to negative infinity
        mask = mask.unsqueeze(-1)  # [B, N, 1]
        clustering_weights = clustering_weights.masked_fill(mask == 0, float('-inf'))

        # Normalize weights
        clustering_weights = F.softmax(clustering_weights, dim=1)  # [B, N, C]

        # Compute cluster embeddings
        cluster_emb = torch.einsum('bnc,bnd->bcd', clustering_weights, node_emb)  # [B, C, D]

        return cluster_emb, clustering_weights


class ClusterSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)  # Add LayerNorm to input
        self.attn = MultiheadSelfAttention(embed_dim=dim, num_heads=heads)  # Self-attention
        self.ln_out = nn.LayerNorm(dim)  # Add LayerNorm to attention output
        self.mlp = nn.Sequential(  # Define MLP
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, cluster_emb):
        '''
        :param cluster_emb: [B, C, D], B is batch size, C is number of clusters, D is feature dimension
        :return: updated_cluster_emb: [B, C, D], attn_matrix: [B, C, C]
        '''
        # 1. Apply LayerNorm to input
        cluster_emb_norm = self.ln_q(cluster_emb)

        # 2. Compute self-attention
        attn_output, attn_matrix = self.attn(cluster_emb_norm)

        # 3. Residual connection
        cluster_emb = cluster_emb + attn_output

        # 4. Apply LayerNorm to residual result
        cluster_emb_norm = self.ln_out(cluster_emb)

        # 5. Pass through MLP
        mlp_output = self.mlp(cluster_emb_norm)

        # 6. Another residual connection
        cluster_emb = cluster_emb + mlp_output

        return cluster_emb, attn_matrix


class NodeClusterReconstruction(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln_q = nn.LayerNorm(dim)  # Add LayerNorm to query
        self.ln_kv = nn.LayerNorm(dim)  # Add LayerNorm to key_value
        self.cross_attn = MultiheadCrossAttention(dim=dim)
        self.ln_out = nn.LayerNorm(dim)  # Add LayerNorm to attention output
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, node_emb, cluster_emb, clustering_weights):
        # 1. Apply LayerNorm to query and key_value
        node_emb_norm = self.ln_q(node_emb)
        cluster_emb_norm = self.ln_kv(cluster_emb)

        # 2. Perform MultiheadCrossAttention
        attn_output, attn_matrix = self.cross_attn(node_emb_norm, cluster_emb_norm, clustering_weights)

        # 3. Residual connection
        node_emb = node_emb + attn_output

        # 4. Apply LayerNorm to residual result
        node_emb_norm = self.ln_out(node_emb)

        # 5. Pass through MLP
        mlp_output = self.mlp(node_emb_norm)

        # 6. Another residual connection
        node_emb = node_emb + mlp_output

        return node_emb, attn_matrix


class ClusterTransformer(nn.Module):
    def __init__(self, dim, num_clusters):
        super().__init__()

        self.num_clusters = num_clusters
        self.clustering = AttentionAwareClustering(dim, num_clusters)
        self.self_attn_cluster = ClusterSelfAttention(dim)
        self.reconstruction = NodeClusterReconstruction(dim)

    def forward(self, node_emb, node_cluster_attention, mask):
        B, N, D = node_emb.shape  # Get batch_size, number of nodes, feature dimension

        # If node_cluster_attention is None, generate a normalized node_cluster_attention with shape [B, N, C], where the sum of cluster weights for each node is 1
        if node_cluster_attention is None:
            # Randomly initialize a [B, N, C] matrix
            node_cluster_attention = torch.randn(B, N, self.num_clusters, device=node_emb.device)  # [B, N, C]
            # Normalize cluster weights for each node using softmax
            node_cluster_attention = F.softmax(node_cluster_attention, dim=-1)  # [B, N, C]
        # 1. Learn clustering through node embeddings
        cluster_emb, clustering_weights = self.clustering(node_emb, node_cluster_attention, mask)
        # 2. Perform self-attention within clusters
        cluster_emb, _ = self.self_attn_cluster(cluster_emb)
        # 3. Map cluster embeddings back to node embeddings through cross-attention
        node_emb, node_cluster_attention = self.reconstruction(node_emb, cluster_emb, clustering_weights)

        return node_emb, node_cluster_attention, clustering_weights


class KMeansTransformer(nn.Module):
    def __init__(self, dim, num_clusters):
        super().__init__()

        self.num_clusters = num_clusters
        self.clustering = AttentionAwareClustering(dim, num_clusters)
        self.self_attn_cluster = ClusterSelfAttention(dim)
        self.reconstruction = NodeClusterReconstruction(dim)

    def forward(self, node_emb, node_cluster_attention, mask):
        B, N, D = node_emb.shape  # Get batch_size, number of nodes, feature dimension

        # If node_cluster_attention is None, generate a normalized node_cluster_attention with shape [B, N, C], where the sum of cluster weights for each node is 1
        if node_cluster_attention is None:
            # Randomly initialize a [B, N, C] matrix
            node_cluster_attention = torch.randn(B, N, self.num_clusters, device=node_emb.device)  # [B, N, C]
            # Normalize cluster weights for each node using softmax
            node_cluster_attention = F.softmax(node_cluster_attention, dim=-1)  # [B, N, C]
        # 1. Learn clustering through node embeddings
        cluster_emb, clustering_weights = self.clustering(node_emb, node_cluster_attention, mask)
        # 2. Perform self-attention within clusters
        cluster_emb, _ = self.self_attn_cluster(cluster_emb)
        # 3. Map cluster embeddings back to node embeddings through cross-attention
        node_emb, node_cluster_attention = self.reconstruction(node_emb, cluster_emb, clustering_weights)

        return node_emb, node_cluster_attention