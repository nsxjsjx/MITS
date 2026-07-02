import os
import torch
from utils.bytecode_to_token import BytecodeOrOpcodeTokenizer


def generate_bytecode_tokens():
    root_dir = '../dataset'
    tokenizer = BytecodeOrOpcodeTokenizer()

    for vul_type in os.listdir(root_dir):
        bytecode_dir = os.path.join(root_dir, vul_type, "bytecode_pure")
        output_dir = os.path.join(root_dir, vul_type, "bytecode_tokens")
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(bytecode_dir):
            if filename.endswith(".sol"):
                input_path = os.path.join(bytecode_dir, filename)
                output_path = os.path.join(output_dir, filename.replace('.sol', '.pt'))

                # 读取纯字节码（只取第一行非空字节码）
                with open(input_path, 'r') as f:
                    bytecode = f.readlines()[0]

                # tokenize
                tokens,padding_mask = tokenizer.tokenize_bytecode(bytecode)

                # 保存
                torch.save(tokens, output_path)

                print(f"[✓] Processed {filename} → {output_path}")


if __name__=='__main__':
    generate_bytecode_tokens()