#!/usr/bin/env python3
"""
Split-CIFAR-100 Experiment Runner

Run BFG and baseline methods on Split-CIFAR-100 benchmark.

Usage:
    python run_cifar100.py --method bfg --k 0.4 --seeds 42 1 2
    python run_cifar100.py --method ewc --ewc_lambda 5000

Author: Hyunjun Kim (KAIST)
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from bfg import create_model, create_method, ContinualLearningModel, ContinualMethod
from bfg.data import get_data_loaders


def train_epoch(
    model: ContinualLearningModel,
    method: ContinualMethod,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    task_id: int,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        loss = method.compute_loss(inputs, targets, task_id)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: ContinualLearningModel,
    test_loaders: List[DataLoader],
    device: torch.device,
    tasks_to_eval: List[int],
) -> List[float]:
    """Evaluate model on specified tasks."""
    model.eval()
    accuracies = []

    with torch.no_grad():
        for task_id in tasks_to_eval:
            correct = 0
            total = 0

            for inputs, targets in test_loaders[task_id]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs, task_id)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)

            acc = 100.0 * correct / total if total > 0 else 0.0
            accuracies.append(acc)

    return accuracies


def compute_metrics(accuracy_matrix: List[List[float]]) -> Dict[str, float]:
    """Compute continual learning metrics from accuracy matrix."""
    n_tasks = len(accuracy_matrix)

    # Average accuracy (final row)
    final_accuracies = accuracy_matrix[-1]
    avg_accuracy = np.mean(final_accuracies)

    # Forgetting
    forgetting = 0.0
    for j in range(n_tasks - 1):
        peak = max(accuracy_matrix[i][j] for i in range(j, n_tasks))
        final = accuracy_matrix[-1][j]
        forgetting += max(0, peak - final)
    forgetting = forgetting / (n_tasks - 1) if n_tasks > 1 else 0.0

    return {
        'average_accuracy': avg_accuracy,
        'forgetting': forgetting,
        'final_accuracies': final_accuracies,
    }


def run_experiment(
    method_name: str,
    config: Dict,
    seeds: List[int],
    n_tasks: int = 10,
    epochs_per_task: int = 50,
    batch_size: int = 64,
    lr: float = 0.001,
    output_dir: str = './results',
    verbose: bool = True,
):
    """Run experiment for all seeds."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("\nLoading Split-CIFAR-100...")
    train_loaders, test_loaders = get_data_loaders(
        'cifar100', n_tasks=n_tasks, batch_size=batch_size
    )
    print(f"Loaded {n_tasks} tasks")

    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED: {seed}")
        print(f"{'='*60}")

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create model
        classes_per_task = 100 // n_tasks
        model = create_model('cifar100', n_tasks=n_tasks, classes_per_task=classes_per_task)
        model = model.to(device)

        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Create method
        method = create_method(method_name, model, device, config)

        # Training loop
        accuracy_matrix = []

        for task_id in range(n_tasks):
            print(f"\nTask {task_id + 1}/{n_tasks}")

            # Before task callback
            method.before_task(task_id, train_loaders[task_id])

            # Train
            for epoch in range(epochs_per_task):
                loss = train_epoch(
                    model, method, train_loaders[task_id],
                    optimizer, task_id, device
                )

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{epochs_per_task}, Loss: {loss:.4f}")

            # After task callback
            method.after_task(task_id, train_loaders[task_id])

            # Evaluate
            accuracies = evaluate(
                model, test_loaders, device,
                tasks_to_eval=list(range(task_id + 1))
            )
            accuracies.extend([0.0] * (n_tasks - len(accuracies)))
            accuracy_matrix.append(accuracies)

            # Print progress
            print(f"  Task {task_id + 1} complete: "
                  f"Current={accuracies[task_id]:.1f}%, "
                  f"Avg={np.mean(accuracies[:task_id + 1]):.1f}%")

            # BFG-specific: print locked fraction
            if hasattr(method, 'get_locked_fraction'):
                print(f"  Locked fraction: {method.get_locked_fraction()*100:.1f}%")

        # Compute metrics
        metrics = compute_metrics(accuracy_matrix)
        print(f"\nSeed {seed} Results:")
        print(f"  Average Accuracy: {metrics['average_accuracy']:.2f}%")
        print(f"  Forgetting: {metrics['forgetting']:.2f}%")

        all_results.append({
            'seed': seed,
            'accuracy_matrix': accuracy_matrix,
            'metrics': metrics,
        })

    # Aggregate results
    avg_acc = np.mean([r['metrics']['average_accuracy'] for r in all_results])
    std_acc = np.std([r['metrics']['average_accuracy'] for r in all_results])
    avg_fgt = np.mean([r['metrics']['forgetting'] for r in all_results])
    std_fgt = np.std([r['metrics']['forgetting'] for r in all_results])

    aggregate = {
        'average_accuracy': {'mean': avg_acc, 'std': std_acc},
        'forgetting': {'mean': avg_fgt, 'std': std_fgt},
    }

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    print(f"Average Accuracy: {avg_acc:.2f} +/- {std_acc:.2f}")
    print(f"Forgetting: {avg_fgt:.2f} +/- {std_fgt:.2f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(
        output_dir,
        f"cifar100_{method_name}_{timestamp}.json"
    )

    results = {
        'config': {
            'method': method_name,
            'dataset': 'cifar100',
            'n_tasks': n_tasks,
            'epochs_per_task': epochs_per_task,
            'batch_size': batch_size,
            'lr': lr,
            'seeds': seeds,
            'method_config': config,
        },
        'seed_results': all_results,
        'aggregate': aggregate,
        'timestamp': timestamp,
    }

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {result_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Split-CIFAR-100 experiments")

    # Method selection
    parser.add_argument('--method', type=str, default='bfg',
                        choices=['naive', 'ewc', 'bfg', 'spg'])

    # Method-specific parameters
    parser.add_argument('--k', type=float, default=0.4,
                        help="BFG lock fraction (default: 0.4)")
    parser.add_argument('--ewc_lambda', type=float, default=5000.0,
                        help="EWC regularization strength (default: 5000)")
    parser.add_argument('--spg_s', type=float, default=200.0,
                        help="SPG mask sharpness (default: 200)")

    # Training parameters
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 1, 2])

    # Output
    parser.add_argument('--output_dir', type=str, default='./results')
    parser.add_argument('--quiet', action='store_true')

    args = parser.parse_args()

    # Build method config
    config = {
        'bfg_k': args.k,
        'ewc_lambda': args.ewc_lambda,
        'spg_s': args.spg_s,
        'fisher_samples': 2000,
    }

    print(f"Running {args.method.upper()} on Split-CIFAR-100")
    print(f"Config: {config}")

    run_experiment(
        method_name=args.method,
        config=config,
        seeds=args.seeds,
        n_tasks=args.n_tasks,
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
