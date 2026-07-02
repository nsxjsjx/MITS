from preprocess.bytecode_to_gv import extract_cfg_all_vuls
from preprocess.bytecode_to_tokens import generate_bytecode_tokens
from preprocess.source_to_bert import sourcecode_to_bert
from preprocess.bytecode_pure_extractor import extract_bytecode_all
from preprocess.gv_to_pyg_data import process_gv
from preprocess.save_op_group import save_opcode_group


def main():
    # bytecode提纯
    extract_bytecode_all()
    # 保存操作码组别情况
    save_opcode_group()
    # 源代码使用bert提取特征并保存
    sourcecode_to_bert()
    # 字节码提取图结构特征并保存
    extract_cfg_all_vuls()
    process_gv()
    # 字节码提取操作码并tokenization后保存
    generate_bytecode_tokens()


if __name__ == '__main__':
    main()
