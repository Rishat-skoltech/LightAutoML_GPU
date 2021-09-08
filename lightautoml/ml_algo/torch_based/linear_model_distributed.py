"""Linear models based on Torch library."""

from copy import deepcopy
from typing import Sequence, Callable, Optional, Union

import dask_cudf

import numpy as np
import cupy as cp

import torch

from scipy import sparse
from cupyx.scipy import sparse as sparse_gpu
from torch import nn
from torch import optim

from ...tasks.losses import TorchLossWrapper
from ...utils.logging import get_logger

from linear_model_cupy import CatLinear, CatLogisticRegression, CatRegression, CatMulticlass

logger = get_logger(__name__)
ArrayOrSparseMatrix = Union[cp.ndarray, sparse_gpu.spmatrix]



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

        all_params = torch.cat([y.view(-1) for (x, y) in self.model.named_parameters() if x != 'bias'])

        penalty = torch.norm(all_params, 2).pow(2) / 2 / n

        return loss + .5 * penalty / c

    def fit_predict(self, data: dask_cudf.DataFrame,
                    y: dask_cudf.DataFrame,
                    weights: Optional[cp.ndarray] = None,
                    data_val: Optional[dask_cudf.DataFrame] = None,
                    y_val: Optional[dask_cudf.DataFrame] = None,
                    weights_val: Optional[cp.ndarray] = None):
        def _train_distributed():
            pass

    def fit(self, data: cp.ndarray, y: cp.ndarray, weights: Optional[np.ndarray] = None,
            data_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None, weights_val: Optional[np.ndarray] = None):
        """Fit method.

        Args:
            data: Data to train.
            y: Train target values.
            weights: Train items weights.
            data_val: Data to validate.
            y_val: Valid target values.
            weights_val: Validation item weights.

        Returns:
            self.

        """
        assert self.model is not None, 'Model should be defined'
        data, data_cat = self._prepare_data(data)
        if len(y.shape) == 1:
            y = y[:, cp.newaxis]
        y = torch.as_tensor(y.astype(cp.float32), device='cuda')
        if weights is not None:
            weights = torch.as_tensor(weights.astype(cp.float32), device='cuda')

        if data_val is None and y_val is None:
            logger.warning('Validation data should be defined. No validation will be performed and C = 1 will be used')
            self._optimize(data, data_cat, y, weights, 1.)

            return self

        data_val, data_val_cat = self._prepare_data(data_val)

        best_score = -np.inf
        best_model = None
        es = 0

        for c in self.cs:
            self._optimize(data, data_cat, y, weights, c)

            val_pred = self._score(data_val, data_val_cat)

            score = self.metric(y_val, val_pred[:,0], weights_val)
            from cuml.metrics import roc_auc_score
            # print(type(y_val.mean()), val_pred.mean(), roc_auc_score(y_val.astype(cp.float64), val_pred.astype(cp.float64)))
            # print("Score is:", score)
            logger.info('Linear model: C = {0} score = {1}'.format(c, score))
            if score > best_score:
                best_score = score
                best_model = deepcopy(self.model)
                es = 0
            else:
                es += 1
            print("BEST MODEL:", best_model)
            if es >= self.early_stopping:
                break
        if best_model is not None:
            print("saving best model...")
            self.model = best_model

        return self

    def _score(self, data: cp.ndarray, data_cat: Optional[cp.ndarray]) -> cp.ndarray:
        """Get predicts to evaluate performance of model.

        Args:
            data: Numeric data.
            data_cat: Categorical data.

        Returns:
            Predicted target values.

        """
        with torch.set_grad_enabled(False):
            self.model.eval()
            preds = cp.asarray(self.model(data, data_cat))

        return preds

    def predict(self, data: cp.ndarray) -> cp.ndarray:
        """Inference phase.

        Args:
            data: Data to test.

        Returns:
            Predicted target values.

        """
        data, data_cat = self._prepare_data(data)

        return self._score(data, data_cat)


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
