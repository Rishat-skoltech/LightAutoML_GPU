"""Set of loss functions for different machine learning algorithms."""

from .base import _valid_str_metric_names
from .cb import CBLoss
from .lgb import LGBLoss
from .sklearn import SKLoss
from .torch import TORCHLoss, TorchLossWrapper
from .cuml import CUMLLoss
from .xgb_gpu import XGBLoss_gpu, XGBLoss_mgpu

__all__ = ['LGBLoss', 'TORCHLoss', 'SKLoss', 'CBLoss', '_valid_str_metric_names', 'TorchLossWrapper', 'CUMLLoss', 'XGBLoss_gpu', 'XGBLoss_mgpu']
