import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteToSourceModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, source_input_dim=768, source_num_layers=2, num_layers=2, dropout=0.5,
                 model_type='lstm', pool_type='mean'):
        super().__init__()
        self.model_type = model_type.lower()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        # self.pos_encoder = PositionalEncoding(max_len=512, hidden_dim=hidden_dim)

        # 池化层
        self.pool_type = pool_type.lower()
        if self.pool_type == 'mean':
            self.pool = lambda x: x.mean(dim=1)  # [B, H]
        elif self.pool_type == 'max':
            self.pool = lambda x: x.max(dim=1).values
        elif self.pool_type == 'cls':
            self.pool = lambda x: x[:, 0, :]
        else:
            raise ValueError(f"Unsupported pool_type: {self.pool_type}")

        vector_proj = []
        in_dim = source_input_dim
        for i in range(source_num_layers):
            vector_proj.append(nn.Linear(in_dim, hidden_dim * 2))
            vector_proj.append(nn.ReLU())
            vector_proj.append(nn.LayerNorm(hidden_dim * 2))
            vector_proj.append(nn.Dropout(dropout))
            in_dim = hidden_dim * 2
        self.vector_proj = nn.Sequential(*vector_proj)

        if self.model_type == 'lstm':
            self.encoder = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True
            )
            self.output_dim = hidden_dim * 2

        elif self.model_type == 'gru':
            self.encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
                bidirectional=True
            )
            self.output_dim = hidden_dim * 2

        elif self.model_type == 'cnn':
            self.conv_layers = nn.Sequential(
                *[
                    nn.Sequential(
                        nn.Conv1d(
                            in_channels=hidden_dim,
                            out_channels=hidden_dim,
                            kernel_size=3,
                            padding=1
                        ),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                    for i in range(num_layers)
                ]
            )
            self.output_dim = hidden_dim

        elif self.model_type == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )
            self.output_dim = hidden_dim

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.fc = nn.Linear(self.output_dim, 1)

        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tokens, vector=None):
        x = self.embedding(tokens)  # [batch_size, seq_len, hidden_dim]
        x = self.dropout(x)

        # 提取序列信息
        if self.model_type in ['lstm', 'gru']:
            # 提取特征的过程 [batch_size, seq_len, hidden_dim * 2]
            x, _ = self.encoder(x)
            # 池化的过程
            x = self.pool(x)
            # 激活的过程
            x = F.relu(x)
        elif self.model_type == 'cnn':
            x = self.pos_encoder(x)
            x = x.permute(0, 2, 1)  # [batch_size, hidden_dim, seq_len]
            x = self.conv_layers(x)  # [batch_size, hidden_dim*2, seq_len]
            x = torch.mean(x, dim=2)  # [batch_size, hidden_dim*2]
        elif self.model_type == 'transformer':
            x = self.pos_encoder(x)
            x = self.encoder(x)  # [batch_size, seq_len, hidden_dim]
            x = x.mean(dim=1)  # mean pooling

        out_byte = self.fc(x).squeeze(1)  # [batch_size]

        # 只有在训练时传入 vector 才计算对齐特征
        if vector is not None:
            # 提取源码特征
            vector_proj = self.vector_proj(vector)
            out_source = self.fc(vector_proj).squeeze(1)
            return x, out_byte, vector_proj, out_source
        else:
            return _, out_byte, _, _

# class PositionalEncoding(nn.Module):
#     def __init__(self, max_len, hidden_dim):
#         super().__init__()
#         self.pos_embedding = nn.Embedding(max_len, hidden_dim)
#
#     def forward(self, x):
#         """
#         x: [batch_size, seq_len, hidden_dim]
#         """
#         seq_len = x.size(1)
#         positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
#         pos_embed = self.pos_embedding(positions)  # [1, seq_len, hidden_dim]
#         return x + pos_embed
