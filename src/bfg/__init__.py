"""
Binary Fisher Gating (BFG) for Continual Learning

A method that uses 1-bit binary masks to achieve hard protection guarantees
with O(1) storage complexity, reducing metadata by 64x compared to EWC.
"""

from .backbone import CNN4, ContinualLearningModel, create_model
from .methods import (
    ContinualMethod,
    NaiveMethod,
    EWCMethod,
    BFGMethod,
    SPGMethod,
    create_method
)
from .data import get_data_loaders, get_cifar100_loaders, get_tinyimagenet_loaders

__version__ = "1.0.0"
__author__ = "Hyunjun Kim"
__email__ = "hyunjun1121@kaist.ac.kr"

__all__ = [
    # Models
    "CNN4",
    "ContinualLearningModel",
    "create_model",
    # Methods
    "ContinualMethod",
    "NaiveMethod",
    "EWCMethod",
    "BFGMethod",
    "SPGMethod",
    "create_method",
    # Data
    "get_data_loaders",
    "get_cifar100_loaders",
    "get_tinyimagenet_loaders",
]
