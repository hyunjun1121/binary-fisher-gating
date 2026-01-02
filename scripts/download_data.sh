#!/bin/bash
# Data Download Script for BFG Experiments
#
# CIFAR-100: Downloaded automatically by torchvision
# TinyImageNet: Requires manual download (see instructions below)

set -e

DATA_DIR="${1:-./data}"
mkdir -p "$DATA_DIR"

echo "=============================================="
echo "BFG Data Download Script"
echo "=============================================="

# CIFAR-100 (automatic via torchvision)
echo ""
echo "[1/2] CIFAR-100"
echo "  - Will be downloaded automatically when running experiments"
echo "  - Location: $DATA_DIR/cifar-100-python/"
echo "  - Size: ~161 MB"

# TinyImageNet (manual download required)
echo ""
echo "[2/2] TinyImageNet"
echo "  - Manual download required due to hosting restrictions"
echo ""
echo "  Option A: Download from Stanford (original source):"
echo "    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip"
echo "    unzip tiny-imagenet-200.zip -d $DATA_DIR"
echo ""
echo "  Option B: Using Kaggle:"
echo "    kaggle datasets download -d akash2sharma/tiny-imagenet"
echo "    unzip tiny-imagenet.zip -d $DATA_DIR"
echo ""
echo "  After download, ensure structure is:"
echo "    $DATA_DIR/tiny-imagenet-200/"
echo "    ├── train/"
echo "    ├── val/"
echo "    ├── test/"
echo "    └── wnids.txt"
echo ""

# Check if TinyImageNet exists
if [ -d "$DATA_DIR/tiny-imagenet-200" ]; then
    echo "[OK] TinyImageNet found at $DATA_DIR/tiny-imagenet-200"
else
    echo "[!] TinyImageNet not found. Please download manually."
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
