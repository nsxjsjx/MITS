import torch
from torch import nn
from torch_geometric.nn import GatedGraphConv, global_mean_pool, global_max_pool, global_add_pool, GlobalAttention, \
    Set2Set, GINConv, GATConv
import torch.nn.functional as F

class GraphModel(nn.Module):
    def __init__(self, vocab_size=165,
                 hidden_dim_graph=256, net_type_graph='GGNN', num_layers_graph=2, pool_type_graph='mean',
                 dropout_rate_graph=0.5):
        super().__init__()

        # 图结点的token的embedding [num_nodes*num_tokens->num_nodes*num_tokens*embedding_dim]
        self.embedding_graph = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim_graph)
        # 图结点的特征提取
        self.net_graph = self.get_net_graph(net_type_graph, hidden_dim_graph, num_layers_graph)
        # 图结点dropout
        self.dropout_graph = nn.Dropout(dropout_rate_graph)
        # 图结点的池化 [num_nodes*embedding_dim->num_graphs*embedding_dim]
        self.pooling_graph = self.get_pooling_graph(pool_type_graph, hidden_dim_graph)

        input_dim_classifier = hidden_dim_graph
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim_graph, hidden_dim_graph),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim_graph),
            nn.Dropout(dropout_rate_graph),
            nn.Linear(hidden_dim_graph, hidden_dim_graph // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim_graph // 2, 1)
        )

        self.process_token_node = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim_graph,  # 输入特征维度
                out_channels=hidden_dim_graph,  # 输出通道数（通常保持一致）
                kernel_size=2,
                padding=1
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def get_net_graph(self, net_type_graph, hidden_dim_graph, num_layers_graph):
        if net_type_graph == 'GGNN':
            return GatedGraphConv(
                out_channels=hidden_dim_graph,
                num_layers=num_layers_graph  # 传播的步数
            )
        elif net_type_graph == 'GAT':
            net = nn.ModuleList()
            for i in range(num_layers_graph):
                net.append(
                    GATConv(in_channels=hidden_dim_graph, out_channels=hidden_dim_graph, heads=1,
                            concat=False)
                )
            return nn.Sequential(*net)

    def get_pooling_graph(self, pool_type, hidden_dim):
        if pool_type == 'mean':
            return global_mean_pool
        elif pool_type == 'max':
            return global_max_pool
        elif pool_type == 'add':
            return global_add_pool
        elif pool_type == 'attn':
            return GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ))
        elif pool_type == 'set2set':
            return Set2Set(hidden_dim, processing_steps=3)
        else:
            raise ValueError(f"Unsupported pool: {pool_type}")
        return None

    def forward(self, data_graph):
        embed_graph = self.embedding_graph(data_graph.x).permute(0, 2, 1)  # [B, D, T]
        feature_graph = self.process_token_node(embed_graph)  # 1D conv: [B, D, T]
        data_graph.x = F.adaptive_max_pool1d(feature_graph, 1).squeeze(-1)  # [B, D]

        info_learn_graph = self.net_graph(data_graph.x, data_graph.edge_index)
        info_learn_graph = self.dropout_graph(info_learn_graph)
        info_learn_graph = self.pooling_graph(info_learn_graph, data_graph.batch)
        info_learn_graph = F.relu(info_learn_graph)

        # 分类
        out = self.classifier(info_learn_graph)
        return out