import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    GATConv,
    SAGEConv,
    GatedGraphConv,
    global_mean_pool
)


def sanitize_edge_index(edge_index, num_nodes, device):
    if edge_index is None or edge_index.numel() == 0:
        idx = torch.arange(num_nodes, device=device)
        return torch.stack([idx, idx], dim=0)
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        idx = torch.arange(num_nodes, device=device)
        return torch.stack([idx, idx], dim=0)
    return edge_index

class TokenEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

    def forward(self, x):
        """
        x: [num_nodes, num_tokens] (LongTensor)
        """
        emb = self.embedding(x)          # [N, T, D]
        node_feat = emb.mean(dim=1)      # [N, D]
        return node_feat


class GNNEncoder(nn.Module):
    def __init__(self, gnn_type, in_dim, hidden_dim, num_layers=2, heads=4):
        super().__init__()
        self.gnn_type = gnn_type
        self.layers = nn.ModuleList()

        if gnn_type == "GCN":
            self.layers.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

        elif gnn_type == "GAT":
            self.layers.append(GATConv(in_dim, hidden_dim // heads, heads=heads))
            for _ in range(num_layers - 1):
                self.layers.append(
                    GATConv(hidden_dim, hidden_dim // heads, heads=heads)
                )

        elif gnn_type == "SAGE":
            self.layers.append(SAGEConv(in_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.layers.append(SAGEConv(hidden_dim, hidden_dim))

        elif gnn_type == "GGNN":
            self.input_proj = nn.Linear(in_dim, hidden_dim)
            self.ggnn = GatedGraphConv(hidden_dim, num_layers)

        else:
            raise ValueError

    def forward(self, x, edge_index, batch):
        edge_index = sanitize_edge_index(edge_index, x.size(0), x.device)

        if self.gnn_type == "GGNN":
            x = self.input_proj(x)
            x = self.ggnn(x, edge_index)
        else:
            for conv in self.layers:
                x = F.relu(conv(x, edge_index))

        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        g = global_mean_pool(x, batch)
        return g


class StructuralAwareAttention(nn.Module):
    def __init__(self, gnn_dim, struct_dim, hidden_dim):
        super().__init__()

        self.query_proj = nn.Linear(gnn_dim, hidden_dim)
        self.context_proj = nn.Linear(struct_dim, hidden_dim)
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, gnn_embeddings, struct_feat):
        """
        gnn_embeddings: [B, 4, D]
        struct_feat:    [B, S]
        """
        B, K, D = gnn_embeddings.size()

        # expand structure features
        struct_context = self.context_proj(struct_feat)   # [B, H]
        struct_context = struct_context.unsqueeze(1)      # [B, 1, H]

        # attention score
        h = torch.tanh(
            self.query_proj(gnn_embeddings) + struct_context
        )  # [B, 4, H]

        scores = self.score_proj(h).squeeze(-1)  # [B, 4]
        alpha = F.softmax(scores, dim=1)         # [B, 4]

        # weighted sum
        fused = torch.sum(
            gnn_embeddings * alpha.unsqueeze(-1),
            dim=1
        )  # [B, D]

        return fused, alpha


class MultiGNN_CFG_Model(nn.Module):
    def __init__(self, vocab_size, token_dim, hidden_dim, struct_dim, num_classes=2):
        super().__init__()

        self.token_encoder = TokenEncoder(vocab_size, token_dim)

        # todo
        self.gcn  = GNNEncoder("GCN",  token_dim, hidden_dim)
        self.gat  = GNNEncoder("GAT",  token_dim, hidden_dim)
        # self.ggnn = GNNEncoder("GGNN", token_dim, hidden_dim)
        self.sage = GNNEncoder("SAGE", token_dim, hidden_dim)

        self.att_fusion = StructuralAwareAttention(
            gnn_dim=hidden_dim,
            struct_dim=struct_dim,
            hidden_dim=hidden_dim
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, data):
        # token-based node encoding
        x = self.token_encoder(data.x)

        edge_index = data.edge_index
        batch = data.batch

        struct_feat = torch.stack([
            data.num_nodes_feat,
            data.avg_degree_feat,
            data.loop_count_feat,
            data.max_path_len_feat
        ], dim=1).to(x.device).float()

        # todo
        g1 = self.gcn(x, edge_index, batch)
        g2 = self.gat(x, edge_index, batch)
        # g3 = self.ggnn(x, edge_index, batch)
        g4 = self.sage(x, edge_index, batch)

        # todo
        gnn_embeds = torch.stack([g1, g2, g4], dim=1)

        fused_graph, att_weight = self.att_fusion(
            gnn_embeds, struct_feat
        )

        logits = self.classifier(fused_graph)
        return logits, att_weight

