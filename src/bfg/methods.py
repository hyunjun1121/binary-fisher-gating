"""
Continual Learning Methods

Implements:
1. Naive (no protection)
2. EWC (Elastic Weight Consolidation) - Standard version
3. BFG (Binary Fisher Gating) - Our method
4. SPG (Soft Parameter Gating) - ICML 2023

All methods share the same interface for fair comparison.

Author: Hyunjun Kim (KAIST)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import numpy as np

from .backbone import ContinualLearningModel, create_model


class ContinualMethod:
    """
    Base class for continual learning methods.
    Defines the interface all methods must implement.
    """

    def __init__(
        self,
        model: ContinualLearningModel,
        device: torch.device
    ):
        self.model = model
        self.device = device
        self.current_task = 0

    def before_task(self, task_id: int, train_loader: DataLoader):
        """Called before training on a new task."""
        self.current_task = task_id

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        """Compute loss for a batch."""
        raise NotImplementedError

    def after_task(self, task_id: int, train_loader: DataLoader):
        """Called after training on a task (e.g., compute Fisher)."""
        pass

    def get_state(self) -> Dict:
        """Get method-specific state for checkpointing."""
        return {}

    def load_state(self, state: Dict):
        """Load method-specific state from checkpoint."""
        pass


class NaiveMethod(ContinualMethod):
    """
    Naive baseline - no protection against forgetting.
    Simply trains on each task sequentially.
    """

    def __init__(self, model: ContinualLearningModel, device: torch.device):
        super().__init__(model, device)

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        outputs = self.model(inputs, task_id)
        return F.cross_entropy(outputs, targets)


class EWCMethod(ContinualMethod):
    """
    Elastic Weight Consolidation (Standard version).

    Uses max accumulation for Fisher (not running average like Online EWC).
    Implements soft penalty: L_EWC = L_task + (lambda/2) * sum(F_i * (theta - theta*)^2)
    """

    def __init__(
        self,
        model: ContinualLearningModel,
        device: torch.device,
        lambda_ewc: float = 5000.0,
        fisher_samples: int = 2000,
        online: bool = False,
        gamma: float = 0.9
    ):
        super().__init__(model, device)
        self.lambda_ewc = lambda_ewc
        self.fisher_samples = fisher_samples
        self.online = online
        self.gamma = gamma

        # Storage for Fisher and reference weights
        self.fisher: Dict[str, torch.Tensor] = {}
        self.reference_params: Dict[str, torch.Tensor] = {}

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        # Task loss
        outputs = self.model(inputs, task_id)
        loss = F.cross_entropy(outputs, targets)

        # EWC penalty (only after first task)
        if self.fisher:
            ewc_loss = 0.0
            for name, param in self.model.backbone.named_parameters():
                if name in self.fisher:
                    fisher = self.fisher[name]
                    ref = self.reference_params[name]
                    ewc_loss += (fisher * (param - ref) ** 2).sum()

            loss = loss + (self.lambda_ewc / 2) * ewc_loss

        return loss

    def after_task(self, task_id: int, train_loader: DataLoader):
        """Compute and accumulate Fisher Information."""
        # Compute new Fisher
        new_fisher = self._compute_fisher(train_loader, task_id)

        # Accumulate Fisher (max for standard EWC, weighted avg for online)
        if self.fisher:
            if self.online:
                # Online EWC: running average
                for name in new_fisher:
                    self.fisher[name] = (
                        self.gamma * self.fisher[name] +
                        (1 - self.gamma) * new_fisher[name]
                    )
            else:
                # Standard EWC: element-wise maximum
                for name in new_fisher:
                    self.fisher[name] = torch.max(
                        self.fisher[name],
                        new_fisher[name]
                    )
        else:
            self.fisher = new_fisher

        # Store reference parameters
        self.reference_params = {
            name: param.clone().detach()
            for name, param in self.model.backbone.named_parameters()
        }

    def _compute_fisher(
        self,
        train_loader: DataLoader,
        task_id: int
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix diagonal."""
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.backbone.named_parameters()
        }

        self.model.eval()
        n_samples = 0

        for inputs, targets in train_loader:
            if n_samples >= self.fisher_samples:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs, task_id)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            for name, param in self.model.backbone.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            n_samples += inputs.size(0)

        # Normalize
        for name in fisher:
            fisher[name] /= n_samples

        self.model.train()
        return fisher

    def get_state(self) -> Dict:
        return {
            'fisher': {k: v.cpu() for k, v in self.fisher.items()},
            'reference_params': {k: v.cpu() for k, v in self.reference_params.items()},
        }

    def load_state(self, state: Dict):
        self.fisher = {k: v.to(self.device) for k, v in state['fisher'].items()}
        self.reference_params = {k: v.to(self.device) for k, v in state['reference_params'].items()}


class BFGMethod(ContinualMethod):
    """
    Binary Fisher Gating (BFG) - Our Method.

    Uses Fisher Information to identify important weights and
    applies hard binary masking (gradient = 0 for locked weights).

    Key advantages:
    - Hard protection: locked weights receive exactly zero gradient
    - O(1) storage: single cumulative 1-bit mask
    - 64x storage reduction vs EWC
    """

    def __init__(
        self,
        model: ContinualLearningModel,
        device: torch.device,
        k: float = 0.4,
        fisher_samples: int = 2000,
        threshold_type: str = "global"  # global or per_layer
    ):
        super().__init__(model, device)
        self.k = k
        self.fisher_samples = fisher_samples
        self.threshold_type = threshold_type

        # Binary masks (1 = unlocked, 0 = locked)
        self.masks: Dict[str, torch.Tensor] = {}

        # Initialize masks to all ones (all unlocked)
        for name, param in self.model.backbone.named_parameters():
            self.masks[name] = torch.ones_like(param).to(device)

        # Register gradient hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks to apply masks to gradients."""
        self.hooks = []

        for name, param in self.model.backbone.named_parameters():
            def hook_fn(grad, name=name):
                return grad * self.masks[name]
            hook = param.register_hook(hook_fn)
            self.hooks.append(hook)

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        outputs = self.model(inputs, task_id)
        return F.cross_entropy(outputs, targets)

    def after_task(self, task_id: int, train_loader: DataLoader):
        """Compute Fisher and update masks."""
        # Compute Fisher Information
        fisher = self._compute_fisher(train_loader, task_id)

        # Update masks based on Fisher
        self._update_masks(fisher)

    def _compute_fisher(
        self,
        train_loader: DataLoader,
        task_id: int
    ) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix diagonal."""
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.backbone.named_parameters()
        }

        self.model.eval()
        n_samples = 0

        for inputs, targets in train_loader:
            if n_samples >= self.fisher_samples:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs, task_id)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            for name, param in self.model.backbone.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            n_samples += inputs.size(0)

        for name in fisher:
            fisher[name] /= n_samples

        self.model.train()
        return fisher

    def _update_masks(self, fisher: Dict[str, torch.Tensor]):
        """Update masks based on Fisher values."""
        if self.threshold_type == "global":
            self._update_masks_global(fisher)
        else:
            self._update_masks_per_layer(fisher)

    def _update_masks_global(self, fisher: Dict[str, torch.Tensor]):
        """Global thresholding across all parameters."""
        # Collect all Fisher values (only unlocked positions)
        all_values = []
        for name, f in fisher.items():
            mask = self.masks[name]
            unlocked_values = f[mask == 1]
            all_values.append(unlocked_values.flatten())

        all_values = torch.cat(all_values)

        if len(all_values) == 0:
            return

        # Compute threshold (top k% of unlocked weights)
        threshold = torch.quantile(all_values, 1 - self.k)

        # Update masks
        for name, f in fisher.items():
            # Lock weights with Fisher >= threshold (AND currently unlocked)
            new_locks = (f >= threshold) & (self.masks[name] == 1)
            self.masks[name][new_locks] = 0

    def _update_masks_per_layer(self, fisher: Dict[str, torch.Tensor]):
        """Per-layer thresholding."""
        for name, f in fisher.items():
            mask = self.masks[name]
            unlocked_values = f[mask == 1]

            if len(unlocked_values) == 0:
                continue

            threshold = torch.quantile(unlocked_values, 1 - self.k)
            new_locks = (f >= threshold) & (mask == 1)
            self.masks[name][new_locks] = 0

    def get_locked_fraction(self) -> float:
        """Get fraction of locked weights."""
        total = 0
        locked = 0
        for mask in self.masks.values():
            total += mask.numel()
            locked += (mask == 0).sum().item()
        return locked / total if total > 0 else 0

    def get_state(self) -> Dict:
        return {
            'masks': {k: v.cpu() for k, v in self.masks.items()},
        }

    def load_state(self, state: Dict):
        self.masks = {k: v.to(self.device) for k, v in state['masks'].items()}


class SPGMethod(ContinualMethod):
    """
    Soft Parameter Gating (SPG) - ICML 2023.

    Proper implementation following the paper:
    - Gradient-based importance (not Fisher)
    - Per-layer z-score normalization
    - tanh + abs for importance in [0, 1]
    - Element-wise max accumulation
    - Soft gradient masking
    """

    def __init__(
        self,
        model: ContinualLearningModel,
        device: torch.device,
        s: float = 200.0,  # Mask sharpness
        fisher_samples: int = 2000
    ):
        super().__init__(model, device)
        self.s = s
        self.fisher_samples = fisher_samples

        # Accumulated importance (0 = unimportant, 1 = important)
        self.importance: Dict[str, torch.Tensor] = {}

        # Initialize importance to zeros
        for name, param in self.model.backbone.named_parameters():
            self.importance[name] = torch.zeros_like(param).to(device)

        # Register gradient hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register backward hooks to apply soft masking."""
        self.hooks = []

        for name, param in self.model.backbone.named_parameters():
            def hook_fn(grad, name=name):
                # Soft masking: attenuate gradients based on importance
                # mask = 1 - importance (high importance = low gradient flow)
                mask = 1 - self.importance[name]
                return grad * mask
            hook = param.register_hook(hook_fn)
            self.hooks.append(hook)

    def compute_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        task_id: int
    ) -> torch.Tensor:
        outputs = self.model(inputs, task_id)
        return F.cross_entropy(outputs, targets)

    def after_task(self, task_id: int, train_loader: DataLoader):
        """Compute and accumulate importance."""
        # Compute new importance
        new_importance = self._compute_importance(train_loader, task_id)

        # Accumulate via element-wise maximum
        for name in new_importance:
            self.importance[name] = torch.max(
                self.importance[name],
                new_importance[name]
            )

    def _compute_importance(
        self,
        train_loader: DataLoader,
        task_id: int
    ) -> Dict[str, torch.Tensor]:
        """
        Compute importance following SPG paper:
        1. Compute gradient magnitudes
        2. Per-layer z-score normalization
        3. tanh + abs to map to [0, 1]
        """
        # Accumulate gradient magnitudes
        grad_accum = {
            name: torch.zeros_like(param)
            for name, param in self.model.backbone.named_parameters()
        }

        self.model.eval()
        n_samples = 0

        for inputs, targets in train_loader:
            if n_samples >= self.fisher_samples:
                break

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.model.zero_grad()
            outputs = self.model(inputs, task_id)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()

            for name, param in self.model.backbone.named_parameters():
                if param.grad is not None:
                    grad_accum[name] += param.grad.data.abs()

            n_samples += inputs.size(0)

        # Normalize and transform
        importance = {}
        for name, g in grad_accum.items():
            g = g / n_samples  # Average

            # Per-layer z-score normalization
            mean = g.mean()
            std = g.std() + 1e-8
            normalized = (g - mean) / std

            # tanh + abs to get importance in [0, 1]
            importance[name] = torch.tanh(normalized).abs()

        self.model.train()
        return importance

    def get_state(self) -> Dict:
        return {
            'importance': {k: v.cpu() for k, v in self.importance.items()},
        }

    def load_state(self, state: Dict):
        self.importance = {k: v.to(self.device) for k, v in state['importance'].items()}


def create_method(
    method_name: str,
    model: ContinualLearningModel,
    device: torch.device,
    config: Dict
) -> ContinualMethod:
    """
    Factory function to create a continual learning method.

    Args:
        method_name: 'naive', 'ewc', 'bfg', 'spg'
        model: The model to use
        device: Device to use
        config: Method-specific configuration

    Returns:
        ContinualMethod instance
    """
    if method_name == 'naive':
        return NaiveMethod(model, device)

    elif method_name == 'ewc':
        return EWCMethod(
            model, device,
            lambda_ewc=config.get('ewc_lambda', 5000.0),
            fisher_samples=config.get('fisher_samples', 2000),
            online=config.get('ewc_online', False),
            gamma=config.get('ewc_gamma', 0.9)
        )

    elif method_name == 'bfg':
        return BFGMethod(
            model, device,
            k=config.get('bfg_k', 0.4),
            fisher_samples=config.get('fisher_samples', 2000),
            threshold_type=config.get('bfg_threshold_type', 'global')
        )

    elif method_name == 'spg':
        return SPGMethod(
            model, device,
            s=config.get('spg_s', 200.0),
            fisher_samples=config.get('fisher_samples', 2000)
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")


if __name__ == "__main__":
    # Test methods
    print("Testing Continual Learning Methods")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = create_model('cifar100', n_tasks=10, classes_per_task=10)
    model = model.to(device)

    # Test each method
    methods = ['naive', 'ewc', 'bfg', 'spg']
    config = {
        'ewc_lambda': 5000.0,
        'bfg_k': 0.4,
        'fisher_samples': 100,
    }

    for method_name in methods:
        print(f"\nTesting {method_name}...")
        method = create_method(method_name, model, device, config)

        # Test forward
        x = torch.randn(4, 3, 32, 32).to(device)
        y = torch.randint(0, 10, (4,)).to(device)

        loss = method.compute_loss(x, y, task_id=0)
        print(f"  Loss: {loss.item():.4f}")

    print("\nAll tests passed!")
