import os
import re


def extract_bytecode_all():
    dataset_root = '../dataset'

    for vul in os.listdir(dataset_root):
        vul_path = os.path.join(dataset_root, vul)
        bytecode_path = os.path.join(vul_path, 'bytecode')
        output_path = os.path.join(vul_path, 'bytecode_pure')

        if not os.path.isdir(bytecode_path):
            continue  # 跳过没有 bytecode 文件夹的目录

        os.makedirs(output_path, exist_ok=True)

        for filename in os.listdir(bytecode_path):
            if not filename.endswith('.sol'):
                continue

            input_file = os.path.join(bytecode_path, filename)
            output_file = os.path.join(output_path, filename)

            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 正则提取 Binary 后的字节码内容
            match = re.search(r'Binary:\s*\n?([0-9a-fA-F]+)', content)
            if match:
                bytecode = match.group(1)
                with open(output_file, 'w', encoding='utf-8') as out_f:
                    out_f.write(bytecode)
            else:
                print(f"[Warning] No bytecode found in {input_file}")


if __name__ == "__main__":
    extract_bytecode_all()
