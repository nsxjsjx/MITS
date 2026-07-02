import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from bytecode_to_sourcecode.bytecode_source_dataset import BytecodeSourceDataset


class BytecodeEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, max_len=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # 可学习的位置编码（也可以改成 sinusoidal）
        self.position_embedding = nn.Embedding(max_len, hidden_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=3
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [B, L]         => token id序列
        attention_mask: [B, L]    => 1表示保留，0表示padding
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 获取token embedding和位置embedding
        token_embed = self.embedding(input_ids)  # [B, L, D]
        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        position_embed = self.position_embedding(position_ids)  # [B, L, D]
        x = token_embed + position_embed  # [B, L, D]

        # Transformer要求输入为 [L, B, D]
        x = x.transpose(0, 1)  # [L, B, D]

        # src_key_padding_mask: [B, L] => True表示padding位置
        padding_mask = attention_mask == 0 if attention_mask is not None else None
        x = self.encoder(x, src_key_padding_mask=padding_mask)  # [L, B, D]
        x = x.transpose(0, 1)  # [B, L, D]

        # 平均池化: [B, L, D] → [B, D]
        pooled = self.pool(x.transpose(1, 2)).squeeze(-1)  # [B, D]
        return pooled


dataset = BytecodeSourceDataset("../dataset")
dataloader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=dataset.collate_fn,
    shuffle=True
)

# 初始化模型和训练器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BytecodeEncoder(vocab_size=len(dataset.vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # 或者 CosineEmbeddingLoss

# 开始训练
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for bytecode_batch, attention_mask,source_vecs in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        if bytecode_batch is None:
            continue

        bytecode_batch = bytecode_batch.to(device)  # [B, L]
        attention_mask = bytecode_batch.to(device)  # [B, L]
        source_vecs = source_vecs.to(device)  # [B, 768]

        optimizer.zero_grad()
        output_vecs = model(bytecode_batch,attention_mask)  # 加padding mask版本
        # output_vecs = model(bytecode_batch,None)  # 不加padding mask版本

        loss = criterion(output_vecs, source_vecs)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")
