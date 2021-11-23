"""Set of loss functions for different machine learning algorithms."""

from .base import _valid_str_metric_names
from .cb import CBLoss
from .lgb import LGBLoss
from .sklearn import SKLoss
from .torch import TORCHLoss
from .torch import TorchLossWrapper
from .torch_gpu import TORCHLoss_gpu
from .cuml import CUMLLoss
from .xgb_gpu import XGBLoss_gpu

__all__ = [
    "LGBLoss",
    "TORCHLoss",
    "SKLoss",
    "CBLoss",
    "_valid_str_metric_names",
    "TorchLossWrapper",
    "TORCHLoss_gpu",
    "CUMLLoss",
    "XGBLoss_gpu",
]
