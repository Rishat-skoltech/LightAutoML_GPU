"""Distributed linear models based on Torch library."""

import logging

from copy import deepcopy
from typing import Callable
from typing import Optional
from typing import Sequence
from typing import Union
from collections import OrderedDict

import numpy as np
import cupy as cp
import dask.array as da
import dask.dataframe as dd
import dask_cudf
import cudf

from joblib import Parallel, delayed

import os

from cupyx.scipy import sparse as sparse_gpu
import torch
import torch.distributed as dist
from torch import nn
from torch import optim

from ...tasks.losses import TorchLossWrapper

from .linear_model_cupy import CatLinear, CatLogisticRegression, CatRegression, CatMulticlass

logger = logging.getLogger(__name__)
ArrayOrSparseMatrix = Union[cp.ndarray, sparse_gpu.spmatrix]


class TorchBasedLinearEstimator:
    """Linear model based on torch L-BFGS solver (distributed GPU version).

    Accepts Numeric + Label Encoded categories or Numeric sparse input.
    """

    def _score(self, model, data: cp.ndarray, data_cat: Optional[cp.ndarray]) -> cp.ndarray:
        """Get predicts to evaluate performance of model.

        Args:
            data: Numeric data.
            data_cat: Categorical data.

        Returns:
            Predicted target values.

        """
        with torch.set_grad_enabled(False):
            model.eval()
            preds = model(data, data_cat)
            preds = cp.asarray(preds)
            if preds.ndim > 1 and preds.shape[1] == 1:
                preds = cp.squeeze(preds)

        return preds

    def _prepare_data_dense(self, data, y, weights, rank):

        if rank is None:
            ind = 0
            size = 1
            device_id = f'cuda:0'
        else:
            ind = rank
            size = torch.cuda.device_count()
            device_id = f'cuda:{rank}'

        if type(data) == cp.ndarray:
            data = cp.copy(data)
        else:
            data = data.compute()
            
        size_base = data.shape[0]//size
        residue = int(data.shape[0]%size)
        offset = size_base*ind + min(ind, residue)

        if type(data) == cp.ndarray:
            data = data[offset:offset+size_base + int(residue>ind), :]
        else:
            data = data.iloc[offset:offset+size_base + int(residue>ind)]


        if y is not None:
            if type(y) != cp.ndarray:
                y = cp.copy(y.compute().values[offset:offset+size_base + int(residue>ind)])
            else:
                y = cp.copy(y[offset:offset+size_base + int(residue>ind)])
            y = torch.as_tensor(y.astype(cp.float32), device=device_id)
        if weights is not None:

            if type(weights) != cp.ndarray:
                weigths = cp.copy(weights.compute().values[offset:offset+size_base+int(residue>ind)])
            else:
                weights = cp.copy(weights[offset:offset+size_base + int(residue>ind)])

            weights = torch.as_tensor(weights.astype(cp.float32), device=device_id)

        if 0 < len(self.categorical_idx) < data.shape[1]:
            # noinspection PyTypeChecker
            data_cat = torch.as_tensor(
                data[self.categorical_idx].values.astype(cp.int32), 
                device=device_id
            )
            data_ = torch.as_tensor(
                data[data.columns.difference(self.categorical_idx)]\
                 .values.astype(cp.float32),
                device=device_id
            )
            return data_, data_cat, y, weights

        elif len(self.categorical_idx) == 0:
            data_ = torch.as_tensor(data.values.astype(cp.float32), device=device_id)
            return data_, None, y, weights

        else:
            data_cat = torch.as_tensor(data.values.astype(cp.int32), device=device_id)
            return None, data_cat

    def __init__(
        self,
        data_size: int,
        categorical_idx: Sequence[int] = (),
        embed_sizes: Sequence[int] = (),
        output_size: int = 1,
        cs: Sequence[float] =(
            .00001,
            .00005,
            .0001,
            .0005,
            .001,
            .005,
            .01,
            .05,
            .1,
            .5,
            1.,
            2.,
            5.,
            7.,
            10.,
            20.
        ),
        max_iter: int = 1000,
        tol: float = 1e-5,
        early_stopping: int = 2,
        loss=Optional[Callable],
        metric=Optional[Callable],
        gpu_ids=None
    ):
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
        self.gpu_ids = gpu_ids

    def _prepare_data(self, data: ArrayOrSparseMatrix, y = None, weights = None, rank: int = None):
        """Prepare data based on input type.

        Args:
            data: Data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        if sparse_gpu.issparse(data):
            raise NotImplementedError("sparse data on multi GPU is not yet supported")

        return self._prepare_data_dense(data, y, weights,  rank)

    def _optimize(self, model, data: torch.Tensor,
                  data_cat: Optional[torch.Tensor], y: torch.Tensor = None,
                  weights: Optional[torch.Tensor] = None, c: float = 1):
        """Optimize single model.

        Args:
            data: Numeric data to train.
            data_cat: Categorical data to train.
            y: Target values.
            weights: Item weights.
            c: Regularization coefficient.

        """
        model.train()
        opt = optim.LBFGS(
            model.parameters(),
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
            output = model(data, data_cat)
            loss = self._loss_fn(model, y.reshape(-1, 1), output, weights, c)
            if loss.requires_grad:
                loss.backward()
            results.append(loss.item())
            return loss
        opt.step(closure)

    def _loss_fn(
        self,
        model,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        weights: Optional[torch.Tensor],
        c: float
    ) -> torch.Tensor:
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
                    weights: Optional[cp.ndarray] = None,
                    data_val: Optional[dask_cudf.DataFrame] = None,
                    y_val: Optional[dask_cudf.DataFrame] = None,
                    weights_val: Optional[cp.ndarray] = None,
                    dev_id = None):

        assert self.model is not None, 'Model should be defined'

        def train_iteration(data, y, weights, rank, c):
            model = deepcopy(self.model)#.to(rank)
            model.to(rank)
            data, data_cat, y, weights = self._prepare_data(data, y, weights, rank)
            self._optimize(model, data, data_cat, y, weights, c)
            return model.state_dict()

        def score_iteration(data_val, y_val, weights_val, rank):
            model = deepcopy(self.model)#.to(rank)
            model.to(rank)
            data_val, data_val_cat, y_val, weights_val = self._prepare_data(data_val, y_val, weights_val, rank)
            val_pred = self._score(model, data_val, data_val_cat)
            score = self.metric(y_val, val_pred, weights_val)
            return score

        es = 0
        best_score = -np.inf
        torch.cuda.cudart().cudaProfilerStart()
        for c in self.cs:
            with Parallel(n_jobs=len(self.gpu_ids), prefer='threads') as p:
                res = p(delayed(train_iteration)
                (data, y, weights, int(device_id), c)
                for device_id in self.gpu_ids)
            new_state_dict = OrderedDict()
            for i, it in enumerate(res):
                if i == 0:
                    for k in it.keys():
                        new_state_dict[k.replace('module.', '')] = it[k].clone().to('cuda:0')
                else:
                    for k in it.keys():
                        new_state_dict[k.replace('module.', '')] += it[k].clone().to('cuda:0')
            for k in new_state_dict.keys():
                new_state_dict[k] = torch.div(new_state_dict[k], float(len(res)))

            self.model.to('cuda').load_state_dict(new_state_dict)

            with Parallel(n_jobs=len(self.gpu_ids), prefer='threads') as p:
                res = p(delayed(score_iteration)
                (data_val, y_val, weights_val, int(device_id))
                for device_id in self.gpu_ids)

            score = 0.0
            for it in res:
                score += it
            score /= len(self.gpu_ids)

            if score > best_score:
                best_score = score
                best_model_params = deepcopy(new_state_dict)
                es = 0
            else:
                es += 1
            if es >= self.early_stopping:
                break

        self.model.to('cuda').load_state_dict(best_model_params)
        torch.cuda.cudart().cudaProfilerStop()
        return self

    def predict(self, data):
        data_num, data_cat, _, _ = self._prepare_data(data)
        res = self._score(self.model, data_num, data_cat)
        return res


class TorchBasedLogisticRegression(TorchBasedLinearEstimator):
    """Linear binary classifier (distributed GPU version)."""

    def __init__(
        self,
        data_size: int,
        categorical_idx: Sequence[int] = (),
        embed_sizes: Sequence[int] = (),
        output_size: int = 1,
        cs: Sequence[float] = (
            .00001,
            .00005,
            .0001,
            .0005,
            .001,
            .005,
            .01,
            .05,
            .1,
            .5,
            1.,
            2.,
            5.,
            7.,
            10.,
            20.
        ),
        max_iter: int = 1000,
        tol: float = 1e-4,
        early_stopping: int = 2,
        loss=Optional[Callable],
        metric=Optional[Callable],
        gpu_ids = None
    ):
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
        super().__init__(data_size, categorical_idx, embed_sizes, output_size, cs, max_iter, tol, early_stopping, loss, metric, gpu_ids)
        self.model = _model(self.data_size - len(self.categorical_idx), self.embed_sizes, self.output_size).cuda()

    def predict(self, data: cp.ndarray, dev_id = None) -> cp.ndarray:
        """Inference phase.

        Args:
            data: data to test.

        Returns:
            predicted target values.

        """
        pred = super().predict(data)
        #if self._binary:
        #    pred = pred[:, 0]
        return pred


class TorchBasedLinearRegression(TorchBasedLinearEstimator):
    """Torch-based linear regressor optimized by L-BFGS (distributed GPU version)."""

    def __init__(
        self,
        data_size: int,
        categorical_idx: Sequence[int] = (),
        embed_sizes: Sequence[int] = (),
        output_size: int = 1,
        cs: Sequence[float] = (
            .00001,
            .00005,
            .0001,
            .0005,
            .001,
            .005,
            .01,
            .05,
            .1,
            .5,
            1.,
            2.,
            5.,
            7.,
            10.,
            20.
        ),
        max_iter: int = 1000,
        tol: float = 1e-4,
        early_stopping: int = 2,
        loss=Optional[Callable],
        metric=Optional[Callable],
        gpu_ids = None
    ):
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
        super().__init__(
            data_size,
            categorical_idx,
            embed_sizes,
            output_size,
            cs,
            max_iter,
            tol,
            early_stopping,
            loss, 
            metric,
            gpu_ids
        )
        self.model = CatRegression(
            self.data_size - len(self.categorical_idx),
            self.embed_sizes, 
            self.output_size
        ).cuda()

    def predict(self, data: cp.ndarray, dev_id = None) -> cp.ndarray:
        """Inference phase.

        Args:
            data: data to test.

        Returns:
            predicted target values.

        """
        return super().predict(data)[:, 0]

