#!/bin/bash
# Run all experiments for BFG paper reproduction
#
# Expected runtime: ~8-12 hours on a single GPU
# Results will be saved to ./results/
#
# Usage:
#   ./scripts/run_all.sh
#   ./scripts/run_all.sh --quick  # Fewer epochs for testing

set -e

# Parse arguments
EPOCHS=50
if [ "$1" == "--quick" ]; then
    EPOCHS=5
    echo "Quick mode: running with $EPOCHS epochs"
fi

OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "BFG Paper Reproduction Experiments"
echo "=============================================="
echo "Epochs per task: $EPOCHS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Split-CIFAR-100 experiments
echo "[1/4] Running Split-CIFAR-100 experiments..."
echo "----------------------------------------------"

echo "  Running BFG..."
python experiments/run_cifar100.py --method bfg --k 0.4 --epochs $EPOCHS --output_dir $OUTPUT_DIR

echo "  Running EWC..."
python experiments/run_cifar100.py --method ewc --ewc_lambda 5000 --epochs $EPOCHS --output_dir $OUTPUT_DIR

echo "  Running SPG..."
python experiments/run_cifar100.py --method spg --spg_s 200 --epochs $EPOCHS --output_dir $OUTPUT_DIR

echo "  Running Naive..."
python experiments/run_cifar100.py --method naive --epochs $EPOCHS --output_dir $OUTPUT_DIR

echo ""
echo "=============================================="
echo "Experiments complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
