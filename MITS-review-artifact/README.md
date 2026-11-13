# MITS Review Artifact

This repository contains the corrected implementation and the predeclared experimental protocols used in the revised manuscript. It is prepared for anonymous peer review and contains no author-identifying metadata, private datasets, trained checkpoints, or credentials.

## Repository structure

- `revision/`: corrected teacher--student implementation, data loading, training, evaluation, and diagnostic utilities.
- `utils/bytecode_to_token.py`: bytecode tokenization utility used by the recorded experiments.
- `experiments/final_ablation/`: final task-specific feature-path and no-distillation controls.
- `experiments/kernel_validation/`: validation-only Conv1D kernel comparison.
- `experiments/pooling_validation/`: validation-only comparison with mean, max, median, and sum pooling.
- `configs/revision.example.yaml`: portable example of the recorded training configuration.
- `protocols/`: split and experiment protocol records, including code hashes.

## Evaluation policy

The five training seeds are 17, 29, 43, 71, and 101. Model, checkpoint, node-encoder, convolution-kernel, and pooling selection use validation F1 only. The test split is not loaded during training or configuration selection. Test evaluation is performed only after the retained configurations and checkpoint hashes have been fixed.

## Data and checkpoints

The repository does not redistribute third-party source datasets, pretrained GraphCodeBERT files, prepared feature caches, or trained checkpoints. Paths in the example configuration are placeholders and must be replaced locally. The split protocol records the reported task counts and grouping policy.

## Environment

The recorded experiments used Python 3.9 and PyTorch 2.0.1 with CUDA 11.8. Additional dependencies include PyTorch Geometric, Transformers, scikit-learn, NumPy, pandas, NetworkX, and PyYAML.

## Reproduction entry points

Core training and evaluation entry points are provided under `revision/`. The three final control packages each include a preflight mode and refuse test loading during validation-only selection. See the corresponding scripts and protocol files for the declared task, seed, and variant grids.
