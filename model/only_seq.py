import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim_seq=256, net_type_seq='lstm', num_layers_seq=2, pool_type_seq='mean', dropout_rate_seq=0.3):
        super().__init__()
        self.embedding_seq = nn.Embedding(vocab_size, hidden_dim_seq, padding_idx=0)
        self.dropout_seq = nn.Dropout(dropout_rate_seq)
        self.net_seq = self.get_net_seq(net_type_seq, hidden_dim_seq, num_layers_seq, dropout_rate_seq)
        self.pooling_seq = self.get_pooling_seq(pool_type_seq.lower())

        self.fc = nn.Linear(hidden_dim_seq*2, 1)

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

    def forward(self, data_seq):
        info_learn_seq = self.embedding_seq(data_seq) # [batch_size, seq_len, hidden_dim]
        info_learn_seq, _ = self.net_seq(info_learn_seq)
        info_learn_seq = self.pooling_seq(info_learn_seq)
        info_learn_seq = F.relu(info_learn_seq)

        out = self.fc(info_learn_seq)  # [batch_size, 1]
        return out.squeeze(1)  # [batch_size]

