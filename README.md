# Binary Fisher Gating (BFG)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/)

Official implementation of **"Binary Fisher Gating for Continual Learning"**.

## Key Results

| Method | Avg. Accuracy | Forgetting | Storage |
|--------|---------------|------------|---------|
| Naive (Fine-tuning) | 20.7% | 64.1% | 0 |
| EWC | 49.3% | 35.2% | 64 bits/weight |
| SPG | 41.3% | 46.8% | 32 bits/weight |
| **BFG (Ours)** | **69.5%** | **3.6%** | **1 bit/weight** |

*Results on Split-CIFAR-100 (10 tasks, 50 epochs/task, 3 seeds)*

## Method Overview

BFG uses **binary masks** (1 bit per weight) to achieve **hard protection** for important weights:

1. **Compute Fisher Information** after each task to identify important weights
2. **Binarize** using global thresholding: lock top-k% most important weights
3. **Hard gating**: locked weights receive exactly zero gradient ($\Delta w = 0$)

**Why it works**: Soft regularization (EWC, SI) only *attenuates* gradients. Even with 99% attenuation, gradients of $0.01 \times g$ accumulate over 50 epochs × 10 tasks to cause significant drift. BFG's hard gating provides *absolute protection*.

## Installation

```bash
git clone https://github.com/hyunjun1121/binary-fisher-gating.git
cd binary-fisher-gating
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- CUDA (optional, for GPU acceleration)

## Quick Start

### Run BFG on Split-CIFAR-100

```bash
python experiments/run_cifar100.py --method bfg --k 0.4 --seeds 42 1 2
```

### Run All Methods (Comparison)

```bash
bash scripts/run_all.sh
```

### Available Methods

| Method | Flag | Key Parameters |
|--------|------|----------------|
| Naive (Fine-tuning) | `--method naive` | - |
| EWC | `--method ewc` | `--ewc_lambda 5000` |
| SPG | `--method spg` | `--spg_s 200` |
| **BFG** | `--method bfg` | `--k 0.4` |

## Experiments

### Main Comparison (Table 1 in paper)

```bash
# Split-CIFAR-100
python experiments/run_cifar100.py --method bfg --k 0.4
python experiments/run_cifar100.py --method ewc --ewc_lambda 5000
python experiments/run_cifar100.py --method spg
python experiments/run_cifar100.py --method naive

# Split-TinyImageNet
python experiments/run_tinyimagenet.py --method bfg --k 0.4
```

### Ablation Studies

```bash
# Global vs Per-layer Thresholding
python experiments/run_ablation.py --type thresholding

# Lock Fraction Sensitivity (k = 0.2, 0.3, 0.4, 0.5, 0.6)
python experiments/run_ablation.py --type lock_sweep

# Quantized EWC (32, 16, 8, 4, 2, 1 bits)
python experiments/run_ablation.py --type quantized_ewc
```

## Project Structure

```
binary-fisher-gating/
├── src/bfg/              # Core implementation
│   ├── methods.py        # BFG, EWC, SPG, Naive
│   ├── backbone.py       # CNN4 architecture
│   └── config.py         # Experiment configuration
├── experiments/          # Experiment scripts
├── scripts/              # Shell scripts for reproduction
├── results/              # Pre-computed results (JSON)
└── paper/                # arXiv submission
```

## Data

### CIFAR-100
Downloaded automatically by torchvision.

### TinyImageNet
Manual download required:
```bash
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ./data
```

## Reproducing Paper Results

All experiments use:
- **50 epochs per task** (exposes cumulative drift in soft methods)
- **3 random seeds** (42, 1, 2) for statistical significance
- **CNN4 backbone** (1.1M parameters, edge-deployment relevant)

Expected runtime per method: ~2-3 hours on a single GPU (NVIDIA RTX 3090).

### Pre-computed Results

Results from our experiments are available in `results/`:
```python
import json
with open('results/cifar100_unified_bfg.json') as f:
    results = json.load(f)
print(f"BFG: {results['aggregate']['average_accuracy']['mean']:.1f}% ± {results['aggregate']['average_accuracy']['std']:.1f}%")
```

## Citation

```bibtex
@article{kim2025bfg,
  title={Binary Fisher Gating for Continual Learning},
  author={Kim, Hyunjun},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contact

- Hyunjun Kim - hyunjun1121@kaist.ac.kr
- KAIST
