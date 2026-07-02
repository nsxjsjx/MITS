import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, GATConv, global_mean_pool, global_max_pool, global_add_pool, \
    GlobalAttention, Set2Set


class MultiModeModel(nn.Module):
    def __init__(self, vocab_size=165,
                 hidden_dim_graph=256, net_type_graph='GGNN', num_layers_graph=2, pool_type_graph='mean',
                 dropout_rate_graph=0.5,
                 hidden_dim_seq=256, net_type_seq='lstm', num_layers_seq=2, pool_type_seq='mean', dropout_rate_seq=0.3,
                 input_dim_source=768, num_layers_source=2, hidden_dim_source=512, dropout_rate_source=0.5):
        super().__init__()
        # 图结点的token的embedding [num_nodes*num_tokens->num_nodes*num_tokens*embedding_dim]
        self.embedding_graph = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_dim_graph)
        # 图结点token序列处理
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
        # 图结点的特征提取
        self.net_graph = self.get_net_graph(net_type_graph, hidden_dim_graph, num_layers_graph)
        # 图结点dropout
        self.dropout_graph = nn.Dropout(dropout_rate_graph)
        # 图结点的池化 [num_nodes*embedding_dim->num_graphs*embedding_dim]
        self.pooling_graph = self.get_pooling_graph(pool_type_graph, hidden_dim_graph)

        self.embedding_seq = nn.Embedding(vocab_size, hidden_dim_seq, padding_idx=0)
        self.dropout_seq = nn.Dropout(dropout_rate_seq)
        self.net_seq = self.get_net_seq(net_type_seq, hidden_dim_seq, num_layers_seq, dropout_rate_seq)
        self.pooling_seq = self.get_pooling_seq(pool_type_seq.lower())

        # 从字节码学习源码特征的网络
        self.net_byte_to_source = nn.Linear(hidden_dim_seq * 2, hidden_dim_source)
        # self.net_byte_to_source = self.get_net_seq(net_type_seq, hidden_dim_seq, num_layers_seq, dropout_rate_seq)
        # self.pooling_byte_to_source = self.get_pooling_seq(pool_type_seq.lower())

        self.net_source = []
        net_source = []
        in_dim = input_dim_source
        for i in range(num_layers_source):
            net_source.append(nn.Linear(in_dim, hidden_dim_source))
            net_source.append(nn.ReLU())
            net_source.append(nn.LayerNorm(hidden_dim_source))
            net_source.append(nn.Dropout(dropout_rate_source))
            in_dim = hidden_dim_source
        self.net_source = nn.Sequential(*net_source)

        input_dim_classifier = hidden_dim_graph
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_dim_classifier, input_dim_classifier // 2),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(input_dim_classifier // 2, 1)
        # )
        # 1. 正常池化的分类器
        self.classifier = nn.Linear(input_dim_classifier, 1)
        # 2. 拼接的分类器
        # self.classifier = nn.Linear(3*input_dim_classifier , 1)

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

    def get_net_seq(self, net_type_seq, hidden_dim_seq, num_layers_seq, dropout_seq):
        if net_type_seq == 'lstm':
            return nn.LSTM(
                input_size=hidden_dim_seq,
                hidden_size=hidden_dim_seq,
                num_layers=num_layers_seq,
                dropout=dropout_seq,
                batch_first=True,
                bidirectional=True
            )
        elif net_type_seq == 'gru':
            return nn.GRU(
                input_size=hidden_dim_seq,
                hidden_size=hidden_dim_seq,
                num_layers=num_layers_seq,
                dropout=dropout_seq,
                batch_first=True,
                bidirectional=True
            )
        elif net_type_seq == 'cnn':
            return nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=hidden_dim_seq,
                            out_channels=hidden_dim_seq,
                            kernel_size=3,
                            padding=1
                        ),
                        nn.ReLU(),
                        nn.Dropout(dropout_seq)
                    )
                    for _ in range(num_layers_seq)
                ]
            )
        elif net_type_seq == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim_seq,
                nhead=2,
                dim_feedforward=hidden_dim_seq,
                dropout=dropout_seq,
                batch_first=True
            )
            return nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers_seq
            )
        else:
            raise ValueError(f"Unsupported model_type: {net_type_seq}")

        return None

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

    def get_pooling_seq(self, pool_type):
        pool = None
        if pool_type == 'mean':
            pool = lambda x: x.mean(dim=1)  # [B, H]
        elif pool_type == 'max':
            pool = lambda x: x.max(dim=1).values
        elif pool_type == 'cls':
            pool = lambda x: x[:, 0, :]
        else:
            raise ValueError(f"Unsupported pooling_seq: {self.pooling_seq}")
        return pool

    def forward(self, data_graph, data_seq, data_source=None):
        # 图
        # embed_graph = self.embedding_graph(data_graph.x).permute(0, 2, 1)
        # feature_graph = self.process_token_node(embed_graph)
        # data_graph.x = F.adaptive_max_pool1d(feature_graph, 1).squeeze(-1)
        # info_learn_graph = self.net_graph(data_graph.x, data_graph.edge_index)
        # info_learn_graph = self.dropout_graph(info_learn_graph)
        # info_learn_graph = self.pooling_graph(info_learn_graph, data_graph.batch)
        # info_learn_graph = F.relu(info_learn_graph)

        # 序列
        info_learn_seq_embedding = self.embedding_seq(data_seq)
        info_learn_seq, _ = self.net_seq(info_learn_seq_embedding)
        info_learn_seq = self.pooling_seq(info_learn_seq)
        info_learn_seq = F.relu(info_learn_seq)

        # 字节码学习源码
        info_learn_byte_to_source = self.net_byte_to_source(info_learn_seq)
        info_learn_byte_to_source = F.relu(info_learn_byte_to_source)
        info_learn_byte_to_source = self.dropout_seq(info_learn_byte_to_source)

        # 融合特征（学生模型）
        fusion_std = torch.stack([info_learn_seq, info_learn_byte_to_source], dim=1)
        # max池化
        features_std_pooled, _ = torch.max(fusion_std, dim=1)
        # 拼接方式
        # features_std_pooled = torch.cat([info_learn_graph, info_learn_seq, info_learn_byte_to_source], dim=1)
        logits_std = self.classifier(features_std_pooled)

        return logits_std.squeeze(1)


if __name__ == '__main__':
    model = MultiModeModel(1)
