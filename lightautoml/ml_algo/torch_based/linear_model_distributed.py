"""Linear models based on Torch library."""

from copy import deepcopy
from typing import Sequence, Callable, Optional, Union

import dask_cudf

import numpy as np
import cupy as cp

import os

import torch

from cupyx.scipy import sparse as sparse_gpu
from torch import nn
from torch import optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from ...tasks.losses import TorchLossWrapper
# from ...utils.logging import get_logger

from .linear_model_cupy import CatLinear, CatLogisticRegression, CatRegression, CatMulticlass

# logger = get_logger(__name__)
ArrayOrSparseMatrix = Union[cp.ndarray, sparse_gpu.spmatrix]


def _setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _cleanup():
    dist.destroy_process_group()


def _prepare_data(data, rank, cat_idx):
    print("Before partitioning: ", data.compute().shape)
    print(f"Rank is {rank}")
    data_part = data.get_partition(rank).compute().values
    print("Shape of computed partition:", data_part.shape)
    device_id = f'cuda:{rank}'
    device_id = 'cuda'
    print("dev id:",device_id)
    if 0 < len(cat_idx) < data.shape[1]:
        # noinspection PyTypeChecker
        data_cat = torch.as_tensor(data_part[:, cat_idx].astype(cp.int64), device=device_id)
        data_ = torch.as_tensor(data_part[:, np.setdiff1d(np.arange(data_part.shape[1]), cat_idx)],
                                device=device_id)
        return data_, data_cat

    elif len(cat_idx) == 0:
        print("In _prepare_data() type of data is", type(data_part), 'shape is', data_part.shape)
        data_ = torch.as_tensor(data_part, device=device_id)
        return data_, None

    else:
        data_cat = torch.as_tensor(data_part.astype(cp.int64), device=device_id)
        return None, data_cat


def _train_distributed(rank, world_size, data, y, model, max_iter_, tol_, weights, cat_idx):
    _setup(rank, world_size)
    data, data_cat = _prepare_data(data, rank, cat_idx)
    print("Type of y:", type(y), y.shape)
    y = torch.as_tensor(y.get_partition(rank).compute().values, device=f'cuda:{rank}')
    if weights is not None:
        weights = torch.as_tensor(weights.astype(cp.float32), device=f'cuda:{rank}')

    dist_model = DDP(model, device_ids=[rank])
    loss_fn = torch.nn.BCELoss()
    dist_model.train()
    opt = optim.LBFGS(
        dist_model.parameters(),
        lr=0.1,
        max_iter=max_iter_,
        tolerance_change=tol_,
        tolerance_grad=tol_,
        line_search_fn='strong_wolfe'
    )

    def closure():
        opt.zero_grad()

        output = dist_model(data, data_cat)
        c = 1
        print(output)
        loss = loss_fn(output, 1.*y.reshape(-1, 1))

        if loss.requires_grad:
            loss.backward()
        return loss

    opt.step(closure)
    _cleanup()


class TorchBasedLinearEstimator:
    """Linear model based on torch L-BFGS solver.

    Accepts Numeric + Label Encoded categories or Numeric sparse input.
    """

    def __init__(self, data_size: int, categorical_idx: Sequence[int] = (), embed_sizes: Sequence[int] = (), output_size: int = 1,
                 cs: Sequence[float] = (.00001, .00005, .0001, .0005, .001, .005, .01, .05, .1, .5, 1., 2., 5., 7., 10., 20.),
                 max_iter: int = 1000, tol: float = 1e-5, early_stopping: int = 2,
                 loss=Optional[Callable], metric=Optional[Callable]):
        """
        Args:
            data_size: Not used.
            categorical_idx: Indices of categorical features.
            embed_sizes: Categorical embedding sizes.
            output_size: Size of output layer.
            cs: Regularization coefficients.
            max_iter: Maximum iterations of L-BFGS.
            tol: Tolerance for the stopping criteria.
            early_stopping: Maximum rounds without improving.
            loss: Loss function. Format: loss(preds, true) -> loss_arr, assume ```reduction='none'```.
            metric: Metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """

        self.data_size = data_size
        self.categorical_idx = categorical_idx
        self.embed_sizes = embed_sizes
        self.output_size = output_size

        assert all([x > 0 for x in cs]), 'All Cs should be greater than 0'

        self.cs = cs
        self.max_iter = max_iter
        self.tol = tol
        self.early_stopping = early_stopping
        self.loss = loss  # loss(preds, true) -> loss_arr, assume reduction='none'
        self.metric = metric  # metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better)

    def _prepare_data(self, data: ArrayOrSparseMatrix):
        """Prepare data based on input type.

        Args:
            data: Data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        # TODO: add sparse support
        # if sparse.issparse(data):
        #     return self._prepare_data_sparse(data)

        return self._prepare_data_dense(data)


    def _prepare_data_dense(self, data: cp.ndarray):
        """Prepare dense matrix.

        Split categorical and numeric features.

        Args:
            data: data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        if 0 < len(self.categorical_idx) < data.shape[1]:
            # noinspection PyTypeChecker
            data_cat = torch.as_tensor(data[:, self.categorical_idx].astype(cp.int64), device='cuda')
            data = torch.as_tensor(data[:, np.setdiff1d(np.arange(data.shape[1]), self.categorical_idx)], device='cuda')
            return data, data_cat

        elif len(self.categorical_idx) == 0:
            data = torch.as_tensor(data, device='cuda')
            return data, None

        else:
            data_cat = torch.as_tensor(data.astype(cp.int64), device='cuda')
            return None, data_cat

    def _optimize(self, data: torch.Tensor, data_cat: Optional[torch.Tensor], y: torch.Tensor = None,
                  weights: Optional[torch.Tensor] = None, c: float = 1):
        """Optimize single model.

        Args:
            data: Numeric data to train.
            data_cat: Categorical data to train.
            y: Target values.
            weights: Item weights.
            c: Regularization coefficient.

        """
        self.model.train()
        opt = optim.LBFGS(
            self.model.parameters(),
            lr=0.1,
            max_iter=self.max_iter,
            tolerance_change=self.tol,
            tolerance_grad=self.tol,
            line_search_fn='strong_wolfe'
        )

        # keep history
        results = []

        def closure():
            opt.zero_grad()

            output = self.model(data, data_cat)
            loss = self._loss_fn(y, output, weights, c)
            if loss.requires_grad:
                loss.backward()
            results.append(loss.item())
            return loss

        opt.step(closure)


    def _loss_fn(self, y_true: torch.Tensor, y_pred: torch.Tensor, weights: Optional[torch.Tensor], c: float) -> torch.Tensor:
        """Weighted loss_fn wrapper.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.
            weights: Item weights.
            c: Regularization coefficients.

        Returns:
            Loss+Regularization value.

        """
        # weighted loss

        loss = self.loss(y_true, y_pred, sample_weight=weights)

        n = y_true.shape[0]
        if weights is not None:
            n = weights.sum()

        all_params = torch.cat([y.view(-1) for (x, y) in model.named_parameters() if x != 'bias'])

        penalty = torch.norm(all_params, 2).pow(2) / 2 / n

        return loss + .5 * penalty / c

    def fit(self, data: dask_cudf.DataFrame,
                    y: dask_cudf.DataFrame,
                    c: int = 1,
                    weights: Optional[cp.ndarray] = None,
                    data_val: Optional[dask_cudf.DataFrame] = None,
                    y_val: Optional[dask_cudf.DataFrame] = None,
                    weights_val: Optional[cp.ndarray] = None):
        # TODO: implement class fit
        return self.fit_predict(data, y, c,weights, data_val, y_val, weights_val)

    def fit_predict(self, data: dask_cudf.DataFrame,
                    y: dask_cudf.DataFrame,
                    c: int = 1,
                    weights: Optional[cp.ndarray] = None,
                    data_val: Optional[dask_cudf.DataFrame] = None,
                    y_val: Optional[dask_cudf.DataFrame] = None,
                    weights_val: Optional[cp.ndarray] = None):

        cat_idx = self.categorical_idx

        assert self.model is not None, 'Model should be defined'

        model = deepcopy(self.model)
        #loss = self._loss_fn
        max_iter = self.max_iter
        tol = self.tol
        world_size = 1
        if True or (data_val is None and y_val is None):
            # logger.warning('Validation data should be defined. No validation will be performed and C = 1 will be used')
            mp.spawn(_train_distributed,
                     args=(world_size, data, y, model, max_iter, tol, weights, cat_idx))

            return self

        #data_val, data_val_cat = self._prepare_data(data_val)

        #best_score = -np.inf
        #best_model = None
        #es = 0

        #for c in self.cs:
        #    self._optimize(data, data_cat, y, weights, c)

        #    val_pred = self._score(data_val, data_val_cat)

        #    score = self.metric(y_val, val_pred[:, 0], weights_val)
        #    from cuml.metrics import roc_auc_score
        #    # print(type(y_val.mean()), val_pred.mean(), roc_auc_score(y_val.astype(cp.float64), val_pred.astype(cp.float64)))
        #    # print("Score is:", score)
        #    logger.info('Linear model: C = {0} score = {1}'.format(c, score))
        #    if score > best_score:
        #        best_score = score
        #        best_model = deepcopy(self.model)
        #        es = 0
        #    else:
        #        es += 1
        #    print("BEST MODEL:", best_model)
        #    if es >= self.early_stopping:
        #        break
        #if best_model is not None:
        #    print("saving best model...")
        #    self.model = best_model

        #return self


class TorchBasedLogisticRegression(TorchBasedLinearEstimator):
    """Linear binary classifier."""

    def __init__(self, data_size: int, categorical_idx: Sequence[int] = (), embed_sizes: Sequence[int] = (), output_size: int = 1,
                 cs: Sequence[float] = (.00001, .00005, .0001, .0005, .001, .005, .01, .05, .1, .5, 1., 2., 5., 7., 10., 20.),
                 max_iter: int = 1000, tol: float = 1e-4, early_stopping: int = 2,
                 loss=Optional[Callable], metric=Optional[Callable]):
        """
        Args:
            data_size: not used.
            categorical_idx: indices of categorical features.
            embed_sizes: categorical embedding sizes.
            output_size: size of output layer.
            cs: regularization coefficients.
            max_iter: maximum iterations of L-BFGS.
            tol: the tolerance for the stopping criteria.
            early_stopping: maximum rounds without improving.
            loss: loss function. Format: loss(preds, true) -> loss_arr, assume reduction='none'.
            metric: metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """
        if output_size == 1:
            _loss = nn.BCELoss
            _model = CatLogisticRegression
            self._binary = True
        else:
            _loss = nn.CrossEntropyLoss
            _model = CatMulticlass
            self._binary = False

        if loss is None:
            loss = TorchLossWrapper(_loss)
        super().__init__(data_size, categorical_idx, embed_sizes, output_size, cs, max_iter, tol, early_stopping, loss, metric)
        self.model = _model(self.data_size - len(self.categorical_idx), self.embed_sizes, self.output_size).cuda()

    def predict(self, data: cp.ndarray) -> cp.ndarray:
        """Inference phase.

        Args:
            data: data to test.

        Returns:
            predicted target values.

        """
        # TODO: make an implementation of predict
        return None
        pred = super().predict(data)
        if self._binary:
            pred = pred[:, 0]
        return pred


class TorchBasedLinearRegression(TorchBasedLinearEstimator):
    """Torch-based linear regressor optimized by L-BFGS."""

    def __init__(self, data_size: int, categorical_idx: Sequence[int] = (), embed_sizes: Sequence[int] = (), output_size: int = 1,
                 cs: Sequence[float] = (.00001, .00005, .0001, .0005, .001, .005, .01, .05, .1, .5, 1., 2., 5., 7., 10., 20.),
                 max_iter: int = 1000, tol: float = 1e-4, early_stopping: int = 2,
                 loss=Optional[Callable], metric=Optional[Callable]):
        """
        Args:
            data_size: used only for super function.
            categorical_idx: indices of categorical features.
            embed_sizes: categorical embedding sizes
            output_size: size of output layer.
            cs: regularization coefficients.
            max_iter: maximum iterations of L-BFGS.
            tol: the tolerance for the stopping criteria.
            early_stopping: maximum rounds without improving.
            loss: loss function. Format: loss(preds, true) -> loss_arr, assume reduction='none'.
            metric: metric function. Format: metric(y_true, y_preds, sample_weight = None) -> float (greater_is_better).

        """
        if loss is None:
            loss = TorchLossWrapper(nn.MSELoss)
        super().__init__(data_size, categorical_idx, embed_sizes, output_size, cs, max_iter, tol, early_stopping, loss, metric)
        self.model = CatRegression(self.data_size - len(self.categorical_idx), self.embed_sizes, self.output_size).cuda()

    def predict(self, data: cp.ndarray) -> cp.ndarray:
        """Inference phase.

        Args:
            data: data to test.

        Returns:
            predicted target values.

        """
        return super().predict(data)[:, 0]
