import os
import torch
import networkx as nx
from tqdm import tqdm

def compute_structural_features(data):
    num_nodes = int(data.num_nodes) if data.num_nodes is not None else 0

    edge_index = getattr(data, "edge_index", None)

    # ===== 安全计算边数 =====
    if edge_index is None:
        num_edges = 0
        edges = []
    elif edge_index.dim() != 2 or edge_index.size(0) != 2:
        num_edges = 0
        edges = []
    else:
        num_edges = edge_index.size(1)
        edges = edge_index.t().tolist()

    # avg degree
    avg_degree = (num_edges * 2 / num_nodes) if num_nodes > 0 else 0.0

    # build nx graph
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # loop count (simple cycles)
    try:
        loop_count = len(list(nx.simple_cycles(G)))
    except Exception:
        loop_count = 0

    # longest path (DAG-safe)
    try:
        if nx.is_directed_acyclic_graph(G):
            max_path_len = nx.dag_longest_path_length(G)
        else:
            max_path_len = 0
    except Exception:
        max_path_len = 0

    return {
        "num_nodes_feat": torch.tensor(float(num_nodes)),
        "avg_degree_feat": torch.tensor(float(avg_degree)),
        "loop_count_feat": torch.tensor(float(loop_count)),
        "max_path_len_feat": torch.tensor(float(max_path_len)),
    }


def preprocess_dataset(vul):
    base_dir = f"dataset/{vul}/cfg_pyg_data"
    files = [f for f in os.listdir(base_dir) if f.endswith(".pt")]

    for f in tqdm(files, desc=f"Preprocessing {vul}"):
        path = os.path.join(base_dir, f)
        save_path = os.path.join(f"dataset/{vul}/cfg_pyg_data_third", f)
        data = torch.load(path, weights_only=False)

        feats = compute_structural_features(data)
        for k, v in feats.items():
            setattr(data, k, v)

        torch.save(data, save_path)


if __name__ == "__main__":
    for vul in ["delegatecall", "reentrancy", "integeroverflow", "timestamp"]:
        preprocess_dataset(vul)