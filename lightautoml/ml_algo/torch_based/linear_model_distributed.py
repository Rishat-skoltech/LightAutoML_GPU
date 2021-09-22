"""Linear models based on Torch library."""

import dask.array as da
import dask.dataframe as dd
import dask_cudf
import cudf

from copy import deepcopy
from typing import Sequence, Callable, Optional, Union
from collections import OrderedDict

import numpy as np
import cupy as cp
from cupyx.scipy import sparse as sparse_gpu

import os

from time import perf_counter

import torch

import torch.distributed as dist
from torch import nn
from torch import optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from ...tasks.losses import TorchLossWrapper
from ...utils.logging import get_logger

from .linear_model_cupy import CatLinear, CatLogisticRegression, CatRegression, CatMulticlass

logger = get_logger(__name__)
ArrayOrSparseMatrix = Union[cp.ndarray, sparse_gpu.spmatrix]


class TorchBasedLinearEstimator:
    """Linear model based on torch L-BFGS solver.

    Accepts Numeric + Label Encoded categories or Numeric sparse input.
    """
    def _train_distributed(self, rank, res_queue, event, world_size, data, y, weights, data_val, y_val, weights_val):

        print(rank, world_size, "rank and world size")
        self._setup(rank, world_size)
        data, data_cat = self._prepare_data_mgpu(data, rank)
        data_val, data_val_cat = self._prepare_data_mgpu(data_val, rank)
        y = torch.as_tensor(y.get_partition(rank).compute().values.astype(cp.float32), device=f'cuda:{rank}')
        #y_val = torch.as_tensor(y_val.get_partition(rank).compute().values.astype(cp.float32), device=f'cuda:{rank}')
        y_val = y_val.get_partition(rank).compute().values.astype(cp.float32)

        if weights is not None:
            weights = torch.as_tensor(weights.astype(cp.float32), device=f'cuda:{rank}')

        best_score = -np.inf
        best_model = self.model
        es = 0
        model = self.model.to(rank)
        dist_model = DDP(model, device_ids=[rank])

        for c in self.cs:
            self._optimize(dist_model, data, data_cat, y, weights, c)
            val_pred = self._score_distributed(dist_model, data_val, data_val_cat)
            score = self.metric(y_val, val_pred, weights_val)
            if score > best_score:
                best_score = score
                if rank == 0:
                    best_model_params = deepcopy(dist_model.state_dict())
                es = 0
            else:
                es += 1
            if es >= self.early_stopping:
                break
        if rank == 0:
            res_queue.put(best_model_params)
            event.wait()
            if res_queue.empty():
                self._cleanup()
        else:
            self._cleanup()

    @staticmethod
    def _score_distributed(dist_model, data_val, data_val_cat):
        with torch.set_grad_enabled(False):
            dist_model.eval()
            preds = cp.asarray(dist_model(data_val, data_val_cat))
        return preds

    def _score(self, data: cp.ndarray, data_cat: Optional[cp.ndarray], rank) -> cp.ndarray:
        """Get predicts to evaluate performance of model.

        Args:
            data: Numeric data.
            data_cat: Categorical data.

        Returns:
            Predicted target values.

        """
        with torch.set_grad_enabled(False):
            model = self.model.to(f'cuda:{rank}')
            model.eval()
            preds = cp.asarray(model(data, data_cat))

        return preds

    def _prepare_data_mgpu(self, data, rank):
        data_part = data.get_partition(rank).compute()#.values.astype(cp.float32)

        device_id = f'cuda:{rank}'
        #device_id = 'cuda'

        if 0 < len(self.categorical_idx) < data.shape[1]:
            # noinspection PyTypeChecker
            data_cat = torch.as_tensor(data_part[self.categorical_idx].values.astype(cp.int32), device=device_id)
            data_ = torch.as_tensor(data_part[data_part.columns.difference(self.categorical_idx)].values.astype(cp.float32),
                                device=device_id)

            return data_, data_cat

        elif len(self.categorical_idx) == 0:
            data_ = torch.as_tensor(data_part.values.astype(cp.float32), device=device_id)
            return data_, None

        else:
            data_cat = torch.as_tensor(data_part.values.astype(cp.int32), device=device_id)
            return None, data_cat

    def _setup(self, rank, world_size):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        # initialize the process group
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    def _cleanup(self):
        dist.destroy_process_group()

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

    def _prepare_data(self, data: ArrayOrSparseMatrix, rank):
        """Prepare data based on input type.

        Args:
            data: Data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        # TODO: add sparse support
        # if sparse.issparse(data):
        #     return self._prepare_data_sparse(data)

        return self._prepare_data_dense(data, rank)


    def _prepare_data_dense(self, data: cp.ndarray, rank):
        """Prepare dense matrix.

        Split categorical and numeric features.

        Args:
            data: data to prepare.

        Returns:
            Tuple (numeric_features, cat_features).

        """
        dev_id = 'cuda:' + str(rank)
        print(dev_id, "dev_id")
        if 0 < len(self.categorical_idx) < data.shape[1]:
            # noinspection PyTypeChecker
            data_cat = torch.as_tensor(data[self.categorical_idx].values.astype(cp.int32), device=dev_id)
            data = torch.as_tensor(data[data.columns.difference(self.categorical_idx)].values.astype(cp.float32), device=dev_id)
            return data, data_cat

        elif len(self.categorical_idx) == 0:
            data = torch.as_tensor(data.values.astype(cp.float32), device=dev_id)
            return data, None

        else:
            data_cat = torch.as_tensor(data.values.astype(cp.int32), device=dev_id)
            return None, data_cat

    def _optimize(self, dist_model, data: torch.Tensor,
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
        dist_model.train()
        opt = optim.LBFGS(
            dist_model.parameters(),
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
            output = dist_model(data, data_cat)
            loss = self._loss_fn(dist_model, y.reshape(-1, 1), output, weights, c)
            if loss.requires_grad:
                loss.backward()
            results.append(loss.item())
            return loss
        opt.step(closure)

    def _loss_fn(self, model, y_true: torch.Tensor, y_pred: torch.Tensor, weights: Optional[torch.Tensor], c: float) -> torch.Tensor:
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
                    weights_val: Optional[cp.ndarray] = None):
        # TODO: implement class fit

        assert self.model is not None, 'Model should be defined'

        world_size = torch.cuda.device_count()

        logger.info("294")
        ctx = mp.get_context('spawn')
        logger.info("295")
        result_queue = ctx.Queue()
        logger.info("296")
        event = ctx.Event()
        logger.info("297")
        ctx = mp.spawn(self._train_distributed, args=(result_queue,
                                                      event,
                                                      world_size,
                                                      data, y,
                                                      weights,
                                                      data_val,
                                                      y_val,
                                                      weights_val),
                       nprocs=world_size,
                       join=False)

        logger.info("298")
        # load model to main process
        while True:
            if not result_queue.empty():
                state_dict = result_queue.get()
                new_state_dict = OrderedDict()

                for k in state_dict.keys():

                    new_state_dict[k.replace('module.','')] = state_dict[k].clone().to('cuda')

                # release CUDA objects
                del state_dict
                event.set()
                break

        logger.info("328")
        # finish multiprocessing
        ctx.join()
        logger.info("331")
        self.model.to('cuda').load_state_dict(new_state_dict)
        return self

    def predict_partition(self, data: cp.ndarray, partition_info = None) -> cp.ndarray:
        """Inference phase.

        Args:
            data: Data to test.

        Returns:
            Predicted target values.

        """
        data_num, data_cat = self._prepare_data(data, partition_info['number'])
        return cudf.DataFrame(self._score(data_num, data_cat, partition_info['number']), index = data.index)

    def predict(self, data):

        res = data.map_partitions(self.predict_partition, meta=cudf.DataFrame(columns=np.arange(self.output_size)).astype(cp.float32) ).persist()
        return res

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
