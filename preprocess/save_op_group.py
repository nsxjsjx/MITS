import json
import os


def save_opcode_group():
    opcode_groups = {
        "STOP": ["STOP", "INVALID", "SELFDESTRUCT"],
        "ARITH": [
            "ADD", "SUB", "MUL", "DIV", "SDIV", "MOD", "SMOD",
            "ADDMOD", "MULMOD", "EXP", "SIGNEXTEND"
        ],
        "COMPARE": [
            "LT", "GT", "SLT", "SGT", "EQ", "ISZERO"
        ],
        "BITWISE": [
            "AND", "OR", "XOR", "NOT", "BYTE", "SHL", "SHR", "SAR"
        ],
        "SHA3": ["SHA3"],
        "ENV": [
            "ADDRESS", "BALANCE", "ORIGIN", "CALLER", "CALLVALUE",
            "CALLDATALOAD", "CALLDATASIZE", "CALLDATACOPY",
            "CODESIZE", "CODECOPY", "EXTCODESIZE", "EXTCODECOPY",
            "RETURNDATASIZE", "RETURNDATACOPY",
            "EXTCODEHASH", "CHAINID", "SELFBALANCE", "BASEFEE"
        ],
        "BLOCK": [
            "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER",
            "DIFFICULTY", "GASLIMIT"
        ],
        "MEMORY": [
            "MLOAD", "MSTORE", "MSTORE8", "MSIZE"
        ],
        "STORAGE": ["SLOAD", "SSTORE"],
        "STACK": [
            "POP",
            *[f"PUSH{i}" for i in range(1, 33)],
            *[f"DUP{i}" for i in range(1, 17)],
            *[f"SWAP{i}" for i in range(1, 17)]
        ],
        "FLOW": ["JUMP", "JUMPI", "PC", "JUMPDEST"],
        "CALL": ["CALL", "CALLCODE", "DELEGATECALL", "STATICCALL", "CREATE", "CREATE2"],
        "RETURN": ["RETURN", "REVERT", "INVALID"],
        "LOG": ["LOG0", "LOG1", "LOG2", "LOG3", "LOG4"],
        "GAS": ["GAS", "GASPRICE"],
        "OTHER": []
    }

    # 保存为 JSON 文件
    output_path = "opcode_groups.json"
    with open(output_path, "w") as f:
        json.dump(opcode_groups, f, indent=4)

    print(f"Opcode group dictionary saved to: {os.path.abspath(output_path)}")
