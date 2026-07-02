# MITS

This repository provides the source code for the paper:

**Token-Aware Multi-Modal Learning for Smart Contract Vulnerability Detection**

MITS is a multi-modal framework for smart contract vulnerability detection. It integrates bytecode-level representations, inferred source-level semantic features, sequential patterns, and graph structures, and uses Conv1D-based intra-node token sequence modeling to enhance node representations.

## Requirements

The code was developed with Python 3.x.

Install dependencies with:

```bash
pip install -r requirements_mits.txt
Repository Structure
MITS/
├── bytecode_to_sourcecode/   # Bytecode-related processing scripts
├── preprocess/               # Data preprocessing scripts
├── model/                    # Model implementation
├── experiments/              # Experimental scripts
├── utils/                    # Utility functions
├── test.py                   # Example evaluation script
├── requirements_mits.txt     # Python dependencies
└── README.md
Dataset

The datasets used in this study are derived from publicly available smart contract vulnerability datasets cited in the paper.

Due to file size and redistribution considerations, the raw datasets are not included in this repository. Please obtain the original datasets from the sources cited in the paper and adjust the data paths in the scripts accordingly.

Usage

Run the preprocessing scripts first:

python preprocess/souce_to_bert.py

Then run the testing or experimental script:

python test.py

Please modify the file paths in the scripts according to your local environment.

Data and Code Availability

The source code used in this study is available in this repository. The datasets are derived from publicly available sources cited in the paper. Processed data and additional implementation details are available from the corresponding author upon reasonable request.

Citation
@misc{mits2026,
  title={Token-Aware Multi-Modal Learning for Smart Contract Vulnerability Detection},
  author={Bi, Lihua and Qu, Yingying and Liu, Zhihui and Jiang, Shunrong},
  year={2026},
  note={Manuscript submitted for publication}
}
