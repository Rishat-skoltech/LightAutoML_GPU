"""Metrics and loss functions for Torch based models."""

from functools import partial
from typing import Callable, Union, Optional, Dict, Any

import torch
from torch import nn

from ..utils import infer_gib
from ..utils_gpu import infer_gib_gpu

from ..common_metric import _valid_str_metric_names
from ..common_metric_gpu import _valid_str_metric_names_gpu

from .base import Loss, MetricFunc


class TorchLossWrapper(nn.Module):
    """Customize PyTorch-based loss.

    Args:
        func: loss to customize. Example: `torch.nn.MSELoss`.
        **kwargs: additional parameters.

    Returns:
        callable loss, uses format (y_true, y_pred, sample_weight).

    """

    def __init__(self, func: Callable, flatten=False, log=False, **kwargs: Any):
        super(TorchLossWrapper, self).__init__()
        self.base_loss = func(reduction='none', **kwargs)
        self.flatten = flatten
        self.log = log

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
        if self.flatten:
            y_true = y_true[:, 0].type(torch.int64)

        if self.log:
            y_pred = torch.log(y_pred)

        outp = self.base_loss(y_pred, y_true)

        if len(outp.shape) == 2:
            outp = outp.sum(dim=1)

        if sample_weight is not None:
            outp = outp * sample_weight
            return outp.mean() / sample_weight.mean()

        return outp.mean()


def torch_rmsle(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
    """Computes Root Mean Squared Logarithmic Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    y_pred = torch.log1p(y_pred)
    y_true = torch.log1p(y_true)

    outp = (y_pred - y_true) ** 2
    if len(outp.shape) == 2:
        outp = outp.sum(dim=1)

    if sample_weight is not None:
        outp = outp * sample_weight
        return outp.mean() / sample_weight.mean()

    return outp.mean()


def torch_quantile(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None,
                   q: float = 0.9):
    """Computes Mean Quantile Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        q: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = err < 0
    err = torch.abs(err)
    err = torch.where(s, err * (1 - q), err * q)

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()


def torch_fair(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None,
               c: float = 0.9):
    """Computes Mean Fair Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        c: metric coefficient.

    Returns:
        metric value.

    """
    x = torch.abs(y_pred - y_true) / c
    err = c ** 2 * (x - torch.log(x + 1))

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()


def torch_huber(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None,
                a: float = 0.9):
    """Computes Mean Huber Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.
        a: metric coefficient.

    Returns:
        metric value.

    """
    err = y_pred - y_true
    s = torch.abs(err) < a
    err = torch.where(s, .5 * (err ** 2), a * torch.abs(err) - .5 * (a ** 2))

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()


def torch_f1(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
    """Computes F1 macro.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    y_true = y_true[:, 0].type(torch.int64)
    y_true_ohe = torch.zeros_like(y_pred)

    y_true_ohe[range(y_true.shape[0]), y_true] = 1
    tp = y_true_ohe * y_pred
    if sample_weight is not None:
        sample_weight = sample_weight.unsqueeze(-1)
        sm = sample_weight.mean()
        tp = (tp * sample_weight).mean(dim=0) / sm
        f1 = (2 * tp) / ((y_pred * sample_weight).mean(dim=0) / sm + (y_true_ohe * sample_weight).mean(dim=0) / sm + 1e-7)

        return - f1.mean()

    tp = torch.mean(tp, dim=0)

    f1 = (2 * tp) / (y_pred.mean(dim=0) + y_true_ohe.mean(dim=0) + 1e-7)

    f1[f1 != f1] = 0

    return - f1.mean()


def torch_mape(y_true: torch.Tensor, y_pred: torch.Tensor, sample_weight: Optional[torch.Tensor] = None):
    """Computes Mean Absolute Percentage Error.

    Args:
        y_true: true target values.
        y_pred: predicted target values.
        sample_weight: specify weighted mean.

    Returns:
        metric value.

    """
    err = (y_true - y_pred) / y_true
    err = torch.abs(err)

    if len(err.shape) == 2:
        err = err.sum(dim=1)

    if sample_weight is not None:
        err = err * sample_weight
        return err.mean() / sample_weight.mean()

    return err.mean()


_torch_loss_dict = {

    'mse': (nn.MSELoss, False, False),
    'mae': (nn.L1Loss, False, False),
    'logloss': (nn.BCELoss, False, False),
    'crossentropy': (nn.CrossEntropyLoss, True, True),
    'rmsle': (torch_rmsle, False, False),
    'mape': (torch_mape, False, False),
    'quantile': (torch_quantile, False, False),
    'fair': (torch_fair, False, False),
    'huber': (torch_huber, False, False),

    'f1': (torch_f1, False, False)

}


class TORCHLoss(Loss):
    """Loss used for PyTorch."""

    def __init__(self, loss: Union[str, Callable], loss_params: Optional[Dict] = None, device: Optional[str] = 'cpu'):
        """

        Args:
            loss: name or callable objective function.
            loss_params: additional loss parameters.

        """
        assert device in ['cpu', 'gpu', 'mgpu'], 'Device must be either CPU or GPU!'
        self.device = device
        self.loss_params = {}
        if loss_params is not None:
            self.loss_params = loss_params

        if loss in ['mse', 'mae', 'logloss', 'crossentropy']:
            self.loss = TorchLossWrapper(*_torch_loss_dict[loss], **self.loss_params)
        elif type(loss) is str:
            self.loss = partial(_torch_loss_dict[loss][0], **self.loss_params)
        else:
            self.loss = partial(loss, **self.loss_params)
            
    def metric_wrapper(self, metric_func: Callable, greater_is_better: Optional[bool],
                       metric_params: Optional[Dict] = None) -> Callable:
        """Customize metric.

        Args:
            metric_func: Callable metric.
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.

        Returns:
            Callable metric.

        """
        if greater_is_better is None:
            if self.device == 'cpu':
                greater_is_better = infer_gib(metric_func)
            else:
                greater_is_better = infer_gib_gpu(metric_func)
            

        m = 2 * float(greater_is_better) - 1

        if metric_params is not None:
            metric_func = partial(metric_func, **metric_params)

        return MetricFunc(metric_func, m, self._bw_func)

    def set_callback_metric(self, metric: Union[str, Callable], greater_is_better: Optional[bool] = None,
                            metric_params: Optional[Dict] = None, task_name: Optional[Dict] = None):
        """Callback metric setter.

        Args:
            metric: Callback metric
            greater_is_better: Whether or not higher value is better.
            metric_params: Additional metric parameters.
            task_name: Name of task.

        Note:
            Value of ``task_name`` should be one of following options:

            -  `'binary'`
            - `'reg'`
            - `'multiclass'`

        """

        assert task_name in ['binary', 'reg', 'multiclass'], 'Incorrect task name: {}'.format(task_name)
        self.metric = metric

        if metric_params is None:
            metric_params = {}

        if type(metric) is str:
            if self.device == 'cpu':
                metric_dict = _valid_str_metric_names[task_name]
            else:
                metric_dict = _valid_str_metric_names_gpu[task_name]
            self.metric_func = self.metric_wrapper(metric_dict[metric], greater_is_better, metric_params)

            self.metric_name = metric
        else:
            # TODO: create check for gpu-compatibility
            self.metric_func = self.metric_wrapper(metric, greater_is_better, metric_params)
            self.metric_name = None
