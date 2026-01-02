"""
Dataset Utilities for Continual Learning Experiments

Supports:
- Split-CIFAR-100 (10 tasks × 10 classes)
- Split-TinyImageNet (10 tasks × 20 classes)

Author: Hyunjun Kim (KAIST)
"""

from typing import Tuple, List
import os
import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np


# =============================================================================
# CIFAR-100
# =============================================================================

def get_cifar100_splits(n_tasks: int = 10, classes_per_task: int = 10) -> List[List[int]]:
    """Generate task splits for CIFAR-100."""
    assert n_tasks * classes_per_task == 100, "Must cover all 100 classes"
    splits = []
    for t in range(n_tasks):
        start_class = t * classes_per_task
        end_class = start_class + classes_per_task
        splits.append(list(range(start_class, end_class)))
    return splits


class CIFAR100TaskDataset(Dataset):
    """CIFAR-100 dataset for a single task with local labels."""

    def __init__(
        self,
        task_id: int,
        train: bool = True,
        n_tasks: int = 10,
        classes_per_task: int = 10,
        data_root: str = './data',
    ):
        self.task_id = task_id
        self.classes_per_task = classes_per_task

        # Transforms
        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4867, 0.4408],
                    std=[0.2675, 0.2565, 0.2761]
                )
            ])

        # Load CIFAR-100
        self.dataset = datasets.CIFAR100(
            root=data_root,
            train=train,
            download=True,
            transform=self.transform
        )

        # Get class indices for this task
        splits = get_cifar100_splits(n_tasks, classes_per_task)
        self.task_classes = splits[task_id]

        # Create mapping: global -> local
        self.global_to_local = {c: i for i, c in enumerate(self.task_classes)}

        # Filter indices
        targets = np.array(self.dataset.targets)
        self.indices = []
        for cls in self.task_classes:
            self.indices.extend(np.where(targets == cls)[0].tolist())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, global_label = self.dataset[real_idx]
        local_label = self.global_to_local[global_label]
        return img, local_label


def get_cifar100_loaders(
    n_tasks: int = 10,
    classes_per_task: int = 10,
    batch_size: int = 64,
    data_root: str = './data',
    num_workers: int = 0,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Get all train and test loaders for Split-CIFAR-100."""
    train_loaders = []
    test_loaders = []

    for task_id in range(n_tasks):
        train_dataset = CIFAR100TaskDataset(
            task_id, train=True, n_tasks=n_tasks,
            classes_per_task=classes_per_task, data_root=data_root
        )
        test_dataset = CIFAR100TaskDataset(
            task_id, train=False, n_tasks=n_tasks,
            classes_per_task=classes_per_task, data_root=data_root
        )

        train_loaders.append(DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        ))
        test_loaders.append(DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        ))

    return train_loaders, test_loaders


# =============================================================================
# TinyImageNet
# =============================================================================

def get_tinyimagenet_splits(n_tasks: int = 10, classes_per_task: int = 20) -> List[List[int]]:
    """Generate task splits for TinyImageNet (200 classes)."""
    assert n_tasks * classes_per_task == 200, "Must cover all 200 classes"
    splits = []
    for t in range(n_tasks):
        start_class = t * classes_per_task
        end_class = start_class + classes_per_task
        splits.append(list(range(start_class, end_class)))
    return splits


class TinyImageNetDataset(Dataset):
    """TinyImageNet dataset for a single task with local labels."""

    def __init__(
        self,
        task_id: int,
        train: bool = True,
        n_tasks: int = 10,
        classes_per_task: int = 20,
        data_root: str = './data/tiny-imagenet-200',
    ):
        self.task_id = task_id
        self.classes_per_task = classes_per_task
        self.data_root = data_root

        # Transforms
        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # Load class list
        wnids_file = os.path.join(data_root, 'wnids.txt')
        with open(wnids_file, 'r') as f:
            self.wnids = [line.strip() for line in f.readlines()]

        # Create wnid -> index mapping
        self.wnid_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}

        # Get class indices for this task
        splits = get_tinyimagenet_splits(n_tasks, classes_per_task)
        self.task_classes = splits[task_id]
        self.task_wnids = [self.wnids[i] for i in self.task_classes]

        # Create mapping: global -> local
        self.global_to_local = {c: i for i, c in enumerate(self.task_classes)}

        # Load samples
        self.samples = []
        if train:
            self._load_train_samples()
        else:
            self._load_val_samples()

    def _load_train_samples(self):
        """Load training samples."""
        train_dir = os.path.join(self.data_root, 'train')
        for wnid in self.task_wnids:
            class_dir = os.path.join(train_dir, wnid, 'images')
            global_label = self.wnid_to_idx[wnid]
            local_label = self.global_to_local[global_label]
            for img_name in os.listdir(class_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, local_label))

    def _load_val_samples(self):
        """Load validation samples."""
        val_dir = os.path.join(self.data_root, 'val')
        # Read val annotations
        annotations_file = os.path.join(val_dir, 'val_annotations.txt')
        img_to_wnid = {}
        with open(annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                img_to_wnid[parts[0]] = parts[1]

        images_dir = os.path.join(val_dir, 'images')
        for img_name, wnid in img_to_wnid.items():
            if wnid in self.task_wnids:
                global_label = self.wnid_to_idx[wnid]
                local_label = self.global_to_local[global_label]
                img_path = os.path.join(images_dir, img_name)
                self.samples.append((img_path, local_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, label


def get_tinyimagenet_loaders(
    n_tasks: int = 10,
    classes_per_task: int = 20,
    batch_size: int = 64,
    data_root: str = './data/tiny-imagenet-200',
    num_workers: int = 0,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """Get all train and test loaders for Split-TinyImageNet."""
    train_loaders = []
    test_loaders = []

    for task_id in range(n_tasks):
        train_dataset = TinyImageNetDataset(
            task_id, train=True, n_tasks=n_tasks,
            classes_per_task=classes_per_task, data_root=data_root
        )
        test_dataset = TinyImageNetDataset(
            task_id, train=False, n_tasks=n_tasks,
            classes_per_task=classes_per_task, data_root=data_root
        )

        train_loaders.append(DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        ))
        test_loaders.append(DataLoader(
            test_dataset, batch_size=batch_size,
            shuffle=False, num_workers=num_workers, pin_memory=True
        ))

    return train_loaders, test_loaders


# =============================================================================
# Factory Function
# =============================================================================

def get_data_loaders(
    dataset: str,
    n_tasks: int = 10,
    batch_size: int = 64,
    data_root: str = './data',
    num_workers: int = 0,
) -> Tuple[List[DataLoader], List[DataLoader]]:
    """
    Factory function to get data loaders for any supported dataset.

    Args:
        dataset: 'cifar100' or 'tinyimagenet'
        n_tasks: Number of tasks
        batch_size: Batch size
        data_root: Root directory for data
        num_workers: Number of workers for DataLoader

    Returns:
        (train_loaders, test_loaders) tuple
    """
    if dataset == 'cifar100':
        classes_per_task = 100 // n_tasks
        return get_cifar100_loaders(
            n_tasks=n_tasks,
            classes_per_task=classes_per_task,
            batch_size=batch_size,
            data_root=data_root,
            num_workers=num_workers,
        )
    elif dataset == 'tinyimagenet':
        classes_per_task = 200 // n_tasks
        tinyimagenet_root = os.path.join(data_root, 'tiny-imagenet-200')
        return get_tinyimagenet_loaders(
            n_tasks=n_tasks,
            classes_per_task=classes_per_task,
            batch_size=batch_size,
            data_root=tinyimagenet_root,
            num_workers=num_workers,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


if __name__ == "__main__":
    print("Testing Data Utilities")
    print("=" * 60)

    # Test CIFAR-100
    print("\nTesting CIFAR-100...")
    train_loaders, test_loaders = get_data_loaders('cifar100', n_tasks=10)
    print(f"  Created {len(train_loaders)} train loaders")
    print(f"  Created {len(test_loaders)} test loaders")

    # Check first batch
    batch_x, batch_y = next(iter(train_loaders[0]))
    print(f"  Batch shape: {batch_x.shape}")
    print(f"  Labels: {batch_y[:5]}")

    print("\nAll tests passed!")
