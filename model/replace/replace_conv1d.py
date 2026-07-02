from torch import nn


class MeanPooling(nn.Module):
    def forward(self, x):
        # x: (B, hidden_dim, seq_len)
        return x.mean(dim=2, keepdim=True)

class MaxPooling(nn.Module):
    def forward(self, x):
        return x.max(dim=2, keepdim=True).values

class SumPooling(nn.Module):
    def forward(self, x):
        return x.sum(dim=2, keepdim=True)

class MedianPooling(nn.Module):
    def forward(self, x):
        return x.median(dim=2, keepdim=True).values

class SelfAttention1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):  # x: (B, C, L)
        x_ = x.transpose(1, 2)        # (B, L, C)

        Q = self.q(x_)                # (B, L, C)
        K = self.k(x_)
        V = self.v(x_)

        attn = (Q @ K.transpose(1, 2)) / (Q.shape[-1]**0.5)  # (B, L, L)
        attn = attn.softmax(dim=-1)

        out = attn @ V                # (B, L, C)
        out = out.transpose(1, 2)     # (B, C, L)
        return out


class LinearProject(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # 对每个 token 做线性，然后平均
        x = self.fc(x.transpose(1,2)).transpose(1,2)
        return x.mean(dim=2, keepdim=True)
