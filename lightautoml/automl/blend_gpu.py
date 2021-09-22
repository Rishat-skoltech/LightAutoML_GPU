"""Blenders."""

from typing import Tuple, Sequence, List, cast, Optional, Callable, Union

from scipy.optimize import minimize_scalar

import numpy as np
import cupy as cp
import dask.array as da
import dask.dataframe as dd
import cudf


from .blend import WeightedBlender

from ..dataset.base import LAMLDataset
from ..dataset.cp_cudf_dataset import CudfDataset, CupyDataset
from ..dataset.daskcudf_dataset import DaskCudfDataset
from ..dataset.roles import NumericRole
from ..pipelines.ml.base import MLPipeline
from ..utils.logging import get_logger

logger = get_logger(__name__)

GpuDataset = Union[CudfDataset, CupyDataset, DaskCudfDataset]

class WeightedBlender_gpu(WeightedBlender):
    """Weighted Blender based on coord descent, optimize task metric directly.

    Weight sum eq. 1.
    Good blender for tabular data,
    even if some predictions are NaN (ex. timeout).
    Model with low weights will be pruned.

    """
    def _get_weighted_pred(self, splitted_preds: Sequence[GpuDataset], wts: Optional[cp.ndarray]) -> GpuDataset:
        length = len(splitted_preds)
        if wts is None:
            wts = cp.ones(length, dtype=np.float32) / length

        weighted_pred = None
        if type(splitted_preds[0]) == DaskCudfDataset:
            dask_array = [x.data.to_dask_array(lengths=True) for x in splitted_preds]
            weighted_pred = da.nansum(da.array([x * w for (x, w) in zip(dask_array, wts)]), axis=0).astype(cp.float32)

            not_nulls = da.sum(da.array([da.logical_not(da.isnan(x).any(axis=1)) * w for (x, w) in zip(dask_array, wts)]),
                               axis=0).astype(cp.float32)
            not_nulls = not_nulls[:, cp.newaxis]

            weighted_pred /= not_nulls
            weighted_pred = da.where(not_nulls == 0, cp.nan, weighted_pred)

            cols = ['WeightedBlend_{0}'.format(x) for x in range(weighted_pred.shape[1])]
            index = splitted_preds[0].data.index
            weighted_pred = dd.from_dask_array(weighted_pred, columns = cols, index = index, meta=cudf.DataFrame())

        else:

            weighted_pred = cp.nansum(cp.array([x.data * w for (x, w) in zip(splitted_preds, wts)]), axis=0).astype(np.float32)

            not_nulls = cp.sum(cp.array([cp.logical_not(cp.isnan(x.data).any(axis=1)) * w for (x, w) in zip(splitted_preds, wts)]),
                               axis=0).astype(np.float32)

            not_nulls = not_nulls[:, cp.newaxis]

            weighted_pred /= not_nulls
            weighted_pred = cp.where(not_nulls == 0, cp.nan, weighted_pred)

        outp = splitted_preds[0].empty()

        cols = ['WeightedBlend_{0}'.format(x) for x in range(weighted_pred.shape[1])]
        outp.set_data(weighted_pred, cols,
                      NumericRole(np.float32, prob=self._outp_prob))

        return outp

    def _get_candidate(self, wts: cp.ndarray, idx: int, value: float):

        candidate = wts.copy()
        sl = cp.arange(wts.shape[0]) != idx
        s = candidate[sl].sum()
        candidate[sl] = candidate[sl] / s * (1 - value)
        candidate[idx] = value

        # this is the part for pipeline pruning
        order = candidate.argsort()
        for idx in order:
            if candidate[idx] < self.max_nonzero_coef:
                candidate[idx] = 0
                candidate /= candidate.sum()
            else:
                break

        return candidate

    def _optimize(self, splitted_preds: Sequence[GpuDataset]) -> cp.ndarray:

        length = len(splitted_preds)
        candidate = cp.ones(length, dtype=np.float32) / length
        best_pred = self._get_weighted_pred(splitted_preds, candidate)

        best_score = self.score(best_pred)
        logger.info('Blending: Optimization starts with equal weights and score {0}'.format(best_score))
        score = best_score
        for _ in range(self.max_iters):
            flg_no_upd = True
            for i in range(len(splitted_preds)):
                if candidate[i] == 1:
                    continue

                obj = self._get_scorer(splitted_preds, i, candidate)
                opt_res = minimize_scalar(obj, method='Bounded', bounds=(0, 1),
                                          options={'disp': False, 'maxiter': self.max_inner_iters})
                w = opt_res.x
                score = -opt_res.fun
                if score > best_score:
                    flg_no_upd = False
                    best_score = score
                    # if w < self.max_nonzero_coef:
                    #     w = 0

                    candidate = self._get_candidate(candidate, i, w)

            logger.info('Blending, iter {0}: score = {1}, weights = {2}'.format(_, score, candidate))

            if flg_no_upd:
                logger.info('No score update. Terminated')
                break

        return candidate

    @staticmethod
    def _prune_pipe(pipes: Sequence[MLPipeline], wts: cp.ndarray,
                    pipe_idx: cp.ndarray) -> Tuple[Sequence[MLPipeline], cp.ndarray]:
        new_pipes = []

        for i in range(max(pipe_idx) + 1):
            pipe = pipes[i]
            weights = wts[cp.array(pipe_idx) == i]

            pipe.ml_algos = [x for (x, w) in zip(pipe.ml_algos, weights) if w > 0]

            new_pipes.append(pipe)

        new_pipes = [x for x in new_pipes if len(x.ml_algos) > 0]
        wts = wts[wts > 0]
        return new_pipes, wts

    def _fit_predict(self, predictions: Sequence[GpuDataset], pipes: Sequence[MLPipeline]
                     ) -> Tuple[GpuDataset, Sequence[MLPipeline]]:
        """Perform coordinate descent.

        Args:
            predictions: Sequence of prediction datasets.
            pipes: Sequence of pipelines.

        Returns:
            Single prediction dataset and Sequence of pruned pipelines.

        Returns:
            Dataset and MLPipeline.

        """
        self._set_metadata(predictions, pipes)
        splitted_preds, _, pipe_idx = cast(List[CudfDataset], self.split_models(predictions))

        wts = self._optimize(splitted_preds)
        splitted_preds = [x for (x, w) in zip(splitted_preds, wts) if w > 0]
        pipes, self.wts = self._prune_pipe(pipes, wts, pipe_idx)

        outp = self._get_weighted_pred(splitted_preds, self.wts)

        return outp, pipes

    def _predict(self, predictions: Sequence[LAMLDataset]) -> LAMLDataset:
        """Simple - weighted average.

        Args:
            predictions: Sequence of predictions.

        Returns:
            Dataset with weighted predictions.

        """
        splitted_preds, _, __ = cast(List[CudfDataset], self.split_models(predictions))
        outp = self._get_weighted_pred(splitted_preds, self.wts)

        return outp
