"""Base classes for machine learning algorithms."""

from abc import ABC, abstractmethod
from copy import copy
from typing import Optional, Tuple, Any, List, cast, Dict, Sequence, Union

import cupy as cp
import numpy as np
import cudf
import dask_cudf
import dask.array as da
import dask.dataframe as dd

from .base import TabularMLAlgo

from copy import copy, deepcopy

from lightautoml.validation.base import TrainValidIterator
from ..dataset.base import LAMLDataset
from ..dataset.cp_cudf_dataset import CupyDataset, CudfDataset
from ..dataset.daskcudf_dataset import DaskCudfDataset
from ..dataset.roles import NumericRole
from ..utils.logging import get_logger
from ..utils.timer import TaskTimer, PipelineTimer

logger = get_logger(__name__)
TabularDatasetGpu = Union[CupyDataset, CudfDataset, DaskCudfDataset]

class TabularMLAlgo_gpu(TabularMLAlgo):
    """Machine learning algorithms that accepts cupy arrays as input."""
    _name: str = 'TabularAlgo_gpu'

    def fit_predict(self, train_valid_iterator: TrainValidIterator) -> CupyDataset:
        """Fit and then predict accordig the strategy that uses train_valid_iterator.

        If item uses more then one time it will
        predict mean value of predictions.
        If the element is not used in training then
        the prediction will be ``cp.nan`` for this item

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Dataset with predicted values.

        """
        logger.info('Start fitting {} ...'.format(self._name))
        self.timer.start()
        assert self.is_fitted is False, 'Algo is already fitted'
        # init params on input if no params was set before
        if self._params is None:
            self.params = self.init_params_on_input(train_valid_iterator)

        # save features names
        self._features = train_valid_iterator.features
        # get metric and loss if None
        self.task = train_valid_iterator.train.task

        # get empty validation data to write prediction
        # TODO: Think about this cast
        val_data = train_valid_iterator.get_validation_data().empty()
        preds_ds = None
        if type(val_data) == DaskCudfDataset:
            preds_ds = cast(DaskCudfDataset, val_data)
        else:
            preds_ds = cast(CupyDataset, val_data.to_cupy())
        ########################################################

        outp_dim = 1
        if self.task.name == 'multiclass':
            if type(val_data) == DaskCudfDataset:
                outp_dim = int(preds_ds.target.max().compute()+1)
            else:
                outp_dim = int(cp.max(preds_ds.target) + 1)
        # save n_classes to infer params
        self.n_classes = outp_dim


        preds_arr = None
        counter_arr = None
        
        def zeros_like_daskcudf(data, shape):
            res = cudf.DataFrame(cp.zeros(shape))
            return res

        if type(val_data) == DaskCudfDataset:
            dat = train_valid_iterator.get_validation_data().data
            preds_arr = dat.map_partitions(zeros_like_daskcudf, (dat.shape[0].compute(), 1)).to_dask_array(lengths=True).persist()
            counter_arr = dat.map_partitions(zeros_like_daskcudf, (dat.shape[0].compute(), 1)).to_dask_array(lengths=True).persist()
        else:
            preds_arr = cp.zeros((train_valid_iterator.get_validation_data().shape[0], outp_dim), dtype=cp.float32)
            counter_arr = cp.zeros((train_valid_iterator.get_validation_data().shape[0], 1), dtype=cp.float32)

        # TODO: Make parallel version later
        for n, (idx, train, valid) in enumerate(train_valid_iterator):
            logger.info('\n===== Start working with fold {} for {} =====\n'.format(n, self._name))

            self.timer.set_control_point()
            model, pred = self.fit_predict_single_fold(train, valid)
            self.models.append(model)

            if type(pred) == dask_cudf.DataFrame or type(pred) == dask_cudf.Series:

                if idx is not None:
                    preds_arr[idx] += pred.to_dask_array(lengths=True).reshape(pred.shape[0].compute(), -1)
                    counter_arr[idx] += 1
                else:
                    preds_arr += pred.to_dask_array(lengths=True).reshape(pred.shape[0].compute(), -1)
                    counter_arr += 1

            else:

                preds_arr[idx] += pred.reshape((pred.shape[0], -1))
                counter_arr[idx] += 1

            self.timer.write_run_info()
            if (n + 1) != len(train_valid_iterator):
                # split into separate cases because timeout checking affects parent pipeline timer
                if self.timer.time_limit_exceeded():
                    logger.warning('Time limit exceeded after calculating fold {0}'.format(n))
                    break

        logger.debug('Time history {0}. Time left {1}'.format(self.timer.get_run_results(), self.timer.time_left))
        if type(val_data) == DaskCudfDataset:
            preds_arr /= da.where(counter_arr == 0, 1, counter_arr)
            preds_arr = da.where(counter_arr == 0, cp.nan, preds_arr)
            #THIS WILL STOP WORKING AFTER PREDS ARE TWO DIMENSIONAL, WHICH THEY SHOULD BE
            preds_arr = dd.from_dask_array(preds_arr, columns=[self._name], meta=cudf.DataFrame())
            
        else:
            preds_arr /= cp.where(counter_arr == 0, 1, counter_arr)
            preds_arr = cp.where(counter_arr == 0, cp.nan, preds_arr)
        preds_ds = self._set_prediction(preds_ds, preds_arr)
        logger.info('{} fitting and predicting completed'.format(self._name))
        return preds_ds

    def predict(self, dataset: TabularDatasetGpu) -> CupyDataset:
        """Mean prediction for all fitted models.

        Args:
            dataset: Dataset used for prediction.

        Returns:
            Dataset with predicted values.

        """
        
        assert self.models != [], 'Should be fitted first.'
        if type(dataset) == DaskCudfDataset:
            #multigpu case is not rewriten from fit_predict yet
            raise NotImplementedError
        else:
            preds_ds = dataset.empty().to_cupy()
            preds_arr = None

        for model in self.models:
            if preds_arr is None:
                preds_arr = self.predict_single_fold(model, dataset)
            else:
                preds_arr += self.predict_single_fold(model, dataset)

        preds_arr /= len(self.models)
        preds_arr = preds_arr.reshape((preds_arr.shape[0], -1))
        preds_ds = self._set_prediction(preds_ds, preds_arr)

        return preds_ds
