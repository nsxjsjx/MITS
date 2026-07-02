import os
from binary_extractor.platforms.ETH.cfg import EthereumCFG
from binary_extractor.analysis.graph import CFGGraph


def extract_cfg_all_vuls(root_path='../dataset'):
    for vul in os.listdir(root_path):
        vul_dir = os.path.join(root_path, vul)
        if not os.path.isdir(vul_dir):
            continue  # 跳过非文件夹

        input_dir = os.path.join(vul_dir, 'bytecode_pure')
        output_dir = os.path.join(vul_dir, 'cfg_gv')
        if not os.path.exists(input_dir):
            continue  # 没有bytecode_pure文件夹就跳过

        os.makedirs(output_dir, exist_ok=True)
        print(f'\n[→] Processing {vul}...')

        for filename in os.listdir(input_dir):
            if not filename.endswith('.sol'):
                continue

            filepath = os.path.join(input_dir, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                bytecode_hex = f.read().strip()

            if not all(c in '0123456789abcdefABCDEF' for c in bytecode_hex):
                print(f"[!] Skipped (invalid bytecode): {filename}")
                continue

            try:
                cfg = EthereumCFG(bytecode_hex)
                graph = CFGGraph(cfg)
                graph.view()

                # 读取生成的 .gv 文件内容
                with open('./graph.cfg.gv', 'r', encoding='utf-8') as f:
                    content = f.read()

                # 构造目标路径
                save_path = os.path.join(output_dir, filename.replace('sol', 'gv'))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                # 写入目标文件
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print(f"[✓] CFG saved: {save_path}")
            except Exception as e:
                print(f"[✗] Error processing {filename}: {e}")


if __name__ == '__main__':
    # 启动批量处理
    extract_cfg_all_vuls()
