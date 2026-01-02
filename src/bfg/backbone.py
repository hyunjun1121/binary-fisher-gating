"""
Neural Network Backbones for Continual Learning

Provides consistent backbone architectures across all methods.
All methods share the same backbone for fair comparison.

Author: Hyunjun Kim (KAIST)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CNN4(nn.Module):
    """
    4-Layer CNN backbone used in BFG paper.

    Architecture:
    - Conv1: 3 -> 32, 3x3, BatchNorm, ReLU, MaxPool
    - Conv2: 32 -> 64, 3x3, BatchNorm, ReLU, MaxPool
    - Conv3: 64 -> 128, 3x3, BatchNorm, ReLU, MaxPool
    - Conv4: 128 -> 256, 3x3, BatchNorm, ReLU, MaxPool
    - FC: 256*spatial -> hidden_dim

    Supports both CIFAR-100 (32x32) and TinyImageNet (64x64).
    Total parameters: ~1.1M (edge-deployment relevant)
    """

    def __init__(
        self,
        input_size: int = 32,  # 32 for CIFAR, 64 for TinyImageNet
        hidden_dim: int = 512,
        in_channels: int = 3
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        # Calculate spatial size after 4 pooling operations
        spatial = input_size // (2 ** 4)  # 32->2, 64->4
        self.flat_dim = 256 * spatial * spatial

        # Fully connected layer
        self.fc = nn.Linear(self.flat_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    def get_feature_dim(self) -> int:
        """Return feature dimension."""
        return self.hidden_dim


class MultiHeadClassifier(nn.Module):
    """
    Multi-head classifier for Task-IL setting.
    Each task has its own classification head.
    """

    def __init__(
        self,
        feature_dim: int,
        n_tasks: int,
        classes_per_task: int
    ):
        super().__init__()
        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task

        # Create heads
        self.heads = nn.ModuleDict({
            str(t): nn.Linear(feature_dim, classes_per_task)
            for t in range(n_tasks)
        })

    def forward(self, features: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward through task-specific head."""
        return self.heads[str(task_id)](features)

    def get_all_heads_params(self):
        """Get parameters from all heads."""
        return self.heads.parameters()


class ContinualLearningModel(nn.Module):
    """
    Base model for continual learning experiments.
    Combines backbone + multi-head classifier.
    """

    def __init__(
        self,
        input_size: int = 32,
        hidden_dim: int = 512,
        n_tasks: int = 10,
        classes_per_task: int = 10
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.n_tasks = n_tasks
        self.classes_per_task = classes_per_task

        # Backbone
        self.backbone = CNN4(
            input_size=input_size,
            hidden_dim=hidden_dim
        )

        # Classifier
        self.classifier = MultiHeadClassifier(
            feature_dim=hidden_dim,
            n_tasks=n_tasks,
            classes_per_task=classes_per_task
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """Forward pass for a specific task."""
        features = self.backbone(x)
        logits = self.classifier(features, task_id)
        return logits

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)

    def get_backbone_params(self):
        """Get backbone parameters (for regularization)."""
        return self.backbone.parameters()

    def get_backbone_named_params(self):
        """Get named backbone parameters."""
        return self.backbone.named_parameters()

    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in model."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.classifier.parameters())
        total = backbone_params + head_params

        return {
            'backbone': backbone_params,
            'heads': head_params,
            'total': total,
        }


def create_model(
    dataset: str,
    n_tasks: int,
    classes_per_task: int,
    hidden_dim: int = 512
) -> ContinualLearningModel:
    """
    Factory function to create model based on dataset.

    Args:
        dataset: 'cifar100' or 'tinyimagenet'
        n_tasks: Number of tasks
        classes_per_task: Classes per task
        hidden_dim: Hidden dimension

    Returns:
        ContinualLearningModel instance
    """
    if dataset == 'cifar100':
        input_size = 32
    elif dataset == 'tinyimagenet':
        input_size = 64
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return ContinualLearningModel(
        input_size=input_size,
        hidden_dim=hidden_dim,
        n_tasks=n_tasks,
        classes_per_task=classes_per_task
    )


if __name__ == "__main__":
    # Test models
    print("Testing Backbone Models")
    print("=" * 60)

    # Test CIFAR-100 model
    model_cifar = create_model('cifar100', n_tasks=10, classes_per_task=10)
    x_cifar = torch.randn(4, 3, 32, 32)
    out = model_cifar(x_cifar, task_id=0)
    print(f"CIFAR-100 model output shape: {out.shape}")
    print(f"CIFAR-100 model params: {model_cifar.count_parameters()}")

    # Test TinyImageNet model
    model_tiny = create_model('tinyimagenet', n_tasks=10, classes_per_task=20)
    x_tiny = torch.randn(4, 3, 64, 64)
    out = model_tiny(x_tiny, task_id=0)
    print(f"TinyImageNet model output shape: {out.shape}")
    print(f"TinyImageNet model params: {model_tiny.count_parameters()}")

    print("\nAll tests passed!")
