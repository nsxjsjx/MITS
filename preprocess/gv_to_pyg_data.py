import os
import glob
import torch
from tqdm import tqdm
from torch_geometric.data import Data
from networkx.drawing.nx_agraph import read_dot
import re

from utils.bytecode_to_token import BytecodeOrOpcodeTokenizer


def extract_opcodes_and_operands(instr: str):
    """
    直接过滤掉地址（如 '0:', '2:', 'a:'），保留 opcode 和立即数
    输入: 字节码字符串
    输出: ['PUSH1', '0x80', 'MSTORE', ...]
    """
    tokens = instr.split()
    return [tok for tok in tokens if not re.match(r'^[0-9a-fA-F]+:$', tok)]


# 特征编码
def encode_instruction(instruction):
    instruction = extract_opcodes_and_operands(instruction)
    tokenizer = BytecodeOrOpcodeTokenizer(max_len=50, group_file='opcode_groups.json')
    return tokenizer.tokenize_opcode(instruction)


def gv_to_pyg_data(gv_path):
    # 读取.gv文件，转换为NetworkX图
    nx_graph = read_dot(gv_path)

    # 节点处理
    node_mapping = {}  # node_id -> new index
    node_features = []
    padding_masks = []
    for i, (node, attr) in enumerate(nx_graph.nodes(data=True)):
        node_mapping[node] = i
        label = attr.get("label", "").replace("\\l", "\n")
        instructions = " ".join(label.strip('"').split("\n")).strip()
        features, padding_mask = encode_instruction(instructions)
        padding_masks.append(padding_mask)
        node_features.append(features)

    x = torch.stack(node_features, dim=0)

    # 边处理
    edge_index = []
    for src, dst in nx_graph.edges():
        if src in node_mapping and dst in node_mapping:
            edge_index.append([node_mapping[src], node_mapping[dst]])

    # if not edge_index:
    #     return None  # 跳过无边图

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    return data


def process_gv():
    output_root = "../dataset"

    # 遍历所有vul类型
    dataset_root = "../dataset"
    vul_dirs = [d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))]

    for vul in tqdm(vul_dirs, desc="Processing vul types"):
        gv_dir = os.path.join(dataset_root, vul, "cfg_gv")
        out_dir = os.path.join(output_root, vul, "cfg_pyg_data")
        os.makedirs(out_dir, exist_ok=True)

        gv_files = glob.glob(os.path.join(gv_dir, "*.gv"))
        for gv_file in gv_files:
            data = gv_to_pyg_data(gv_file)
            if data is None:
                continue
            base_name = os.path.splitext(os.path.basename(gv_file))[0]
            torch.save(data, os.path.join(out_dir, base_name + ".pt"))

if __name__ == '__main__':
    process_gv()
