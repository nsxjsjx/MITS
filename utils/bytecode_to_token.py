import json
from typing import List, Tuple
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F

import torch
from pyevmasm import disassemble_all


def load_opcode_groups(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class BytecodeOrOpcodeTokenizer:
    def __init__(self, max_len=512, group_file='../preprocess/opcode_groups.json'):
        # 预定义特殊token
        self.special_tokens = {
            '[PAD]': 0,
            '[CLS]': 1,
            '[SEP]': 2,
            '[UNK]': 3,
            '[MASK]': 4
        }

        # EVM操作码语义分类
        self.opcode_groups = load_opcode_groups(group_file)

        # 构建词汇表
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.max_len = max_len

    def _build_vocab(self):
        """构建层次化词汇表"""
        vocab = defaultdict(int)

        # 1. 添加特殊token
        vocab.update(self.special_tokens)

        # 2. 添加EVM操作码（按语义分组）
        next_id = len(self.special_tokens)
        for group_name, group_ops in self.opcode_groups.items():
            for op in group_ops:
                vocab[f'[OP_{group_name}_{op}]'] = next_id
                next_id += 1

        # 3. 添加常见参数模式
        for i in range(0, 256, 16):  # 16为步长减少稀疏性
            vocab[f'[ARG_{i}]'] = next_id
            next_id += 1

        return vocab

    def tokenize_bytecode(self, hex_bytecode: str) -> torch.Tensor:
        """将十六进制字节码转换为token序列"""
        # 1. 解码字节码
        byte_seq = bytes.fromhex(hex_bytecode)

        # 2. 语义化分段
        tokens = [self.vocab['[CLS]']]
        try:
            instructions = disassemble_all(byte_seq)

            for instr in instructions:
                op = instr.name
                operand = instr.operand  # None or int/bytes

                # 处理操作码语义分组
                group = next(
                    (g for g, ops in self.opcode_groups.items() if op in ops),
                    'OTHER'
                )

                # 添加操作码 token
                op_token = self.vocab.get(f'[OP_{group}_{op}]', self.vocab['[UNK]'])
                tokens.append(op_token)

                # 处理参数：有些指令带参数（如 PUSH、CALL 等）
                if operand is not None:
                    if isinstance(operand, int):
                        param_token = f'[ARG_{operand // 16 * 16}]'
                        tokens.append(self.vocab.get(param_token, self.vocab['[UNK]']))

                    elif isinstance(operand, bytes):
                        # 将每个字节做粗粒度编码
                        for b in operand:
                            param_token = f'[ARG_{b // 16 * 16}]'
                            tokens.append(self.vocab.get(param_token, self.vocab['[UNK]']))

                    else:
                        # 若参数类型为未知格式
                        tokens.append(self.vocab['[UNK]'])

        except Exception as e:
            print(f"Disassembly failed: {e}")
            tokens.append(self.vocab['[UNK]'])

        # 3. 截断/填充
        tokens = tokens[:self.max_len - 1] + [self.vocab['[SEP]']]  # 留位置给[SEP]
        padding_len = self.max_len - len(tokens)
        tokens += [self.vocab['[PAD]']] * padding_len

        # 构建 attention mask：非 PAD 为 1，PAD 为 0
        attention_mask = [1] * (self.max_len - padding_len) + [0] * padding_len

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def tokenize_opcode(self, tokens: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将每个 token（无论是操作码还是立即数）都 tokenize 成一个 vocab id
        """
        ids = [self.vocab['[CLS]']]

        for tok in tokens:
            # 是否是操作码，并尝试分组
            group = next(
                (g for g, ops in self.opcode_groups.items() if tok in ops),
                None
            )

            if group:
                token_id = self.vocab.get(f'[OP_{group}_{tok}]', self.vocab['[UNK]'])
            else:
                # 判断是否为立即数形式
                if tok.startswith('0x'):
                    try:
                        b = int(tok, 16)
                        bucket = b // 16 * 16
                        arg_token = f'[ARG_{bucket}]'
                        token_id = self.vocab.get(arg_token, self.vocab['[UNK]'])
                    except ValueError:
                        token_id = self.vocab['[UNK]']
                else:
                    # 普通未知 token
                    token_id = self.vocab.get(tok, self.vocab['[UNK]'])

            ids.append(token_id)

        # 截断 + 添加 [SEP]
        ids = ids[:self.max_len - 1] + [self.vocab['[SEP]']]
        padding_len = self.max_len - len(ids)
        ids += [self.vocab['[PAD]']] * padding_len

        # attention mask: 非 PAD 为 1，PAD 为 0
        attn_mask = [1] * (self.max_len - padding_len) + [0] * padding_len

        return torch.tensor(ids, dtype=torch.long), torch.tensor(attn_mask, dtype=torch.long)


if __name__ == '__main__':
    node_data=["PUSH1","0x80",'PUSH1' 'Ox40','MSTORE','PUSH1','Ox4','CALLDATASIZE','LT','PUSH1', 'Ox3f','JUMP1']

    token_layer = BytecodeOrOpcodeTokenizer(50)
    node_token,node_mask=token_layer.tokenize_opcode(node_data)

    data=[]
    data.append(node_token)
    x = torch.stack(data, dim=0)

    embedding_layer=nn.Embedding(num_embeddings=165, embedding_dim=512)
    data_embedding=embedding_layer(x).permute(0, 2, 1)

    aggre_layer=nn.Sequential(
            nn.Conv1d(
                in_channels=512,  # 输入特征维度
                out_channels=512,  # 输出通道数（通常保持一致）
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    agrre=aggre_layer(data_embedding)

    res = F.adaptive_max_pool1d(agrre, 1).squeeze(-1)

    print(res)
