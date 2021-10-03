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

import torch

from .base import TabularMLAlgo
from joblib import Parallel, delayed

from copy import copy, deepcopy

from lightautoml.validation.base import TrainValidIterator
from ..dataset.base import LAMLDataset
from ..dataset.cp_cudf_dataset import CupyDataset, CudfDataset
from ..dataset.daskcudf_dataset import DaskCudfDataset
from ..dataset.roles import NumericRole
from ..utils.logging import get_logger
from ..utils.timer import TaskTimer, PipelineTimer

import torch

logger = get_logger(__name__)
TabularDatasetGpu = Union[CupyDataset, CudfDataset, DaskCudfDataset]

class TabularMLAlgo_gpu(TabularMLAlgo):
    """Machine learning algorithms that accepts cupy arrays as input."""
    _name: str = 'TabularAlgo_gpu'
    _parallel_folds: bool = True


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

        #preds_ds = None
        #if type(val_data) == DaskCudfDataset:
        #    preds_ds = cast(DaskCudfDataset, val_data)
        #else:
        preds_ds = cast(CupyDataset, val_data.to_cupy())
        ########################################################
        outp_dim = 1
        if self.task.name == 'multiclass':
            if type(val_data) == DaskCudfDataset:
                outp_dim = int(val_data.target.max().compute()+1)
            else:
                outp_dim = int(cp.max(val_data.target) + 1)
        # save n_classes to infer params
        self.n_classes = outp_dim


        '''preds_arr = None
        counter_arr = None
        
        def zeros_like_daskcudf(data, dim):
            shape = (data.shape[0], dim)
            res = cudf.DataFrame(cp.zeros(shape))
            return res

        if type(val_data) == DaskCudfDataset:
            dat = train_valid_iterator.get_validation_data().data
            preds_arr = dat.map_partitions(zeros_like_daskcudf, outp_dim, meta=cudf.DataFrame(columns=np.arange(outp_dim)).astype(cp.float32)).to_dask_array(lengths=True).persist()
            counter_arr = dat.map_partitions(zeros_like_daskcudf, 1, meta=cudf.DataFrame(columns=[0]).astype(cp.float32)).to_dask_array(lengths=True).persist()
        else:'''
        preds_arr = cp.zeros((train_valid_iterator.get_validation_data().shape[0], outp_dim), dtype=cp.float32)
        counter_arr = cp.zeros((train_valid_iterator.get_validation_data().shape[0], 1), dtype=cp.float32)

        if self._parallel_folds:

            def perform_iterations(fit_predict_single_fold, train_valid, ind, dev_id):
                models = []
                preds = []
                #cp.cuda.runtime.setDevice(dev_id)
                #torch.cuda.set_device(f'cuda:{dev_id}')
                for n in ind:
                    (idx, train, valid) = train_valid[n]
                    logger.info('\n===== Start working with fold {} for {} (par) =====\n'.format(n,self._name))
                    model, pred = fit_predict_single_fold(train, valid, dev_id)
                    models.append(model)
                    preds.append(pred)
                return models, preds

            n_parts = torch.cuda.device_count()
            
            print("Number of GPUs:", n_parts)
            n_folds = len(train_valid_iterator)
            inds = np.array_split(np.arange(n_folds), n_parts)
            inds = [x for x in inds if len(x) > 0]
            device_ids = np.arange(n_parts)#np.zeros(n_parts, dtype='int')
            res = None
            models = []
            preds = []

            train_valid_iterator.train = train_valid_iterator.train.to_cudf()
            
            #with Parallel(n_jobs=n_parts, prefer='processes', 
            #              backend='loky', max_nbytes=None) as p:
            with Parallel(n_jobs=n_parts, prefer='threads') as p: 
                res = p(delayed(perform_iterations)
                (self.fit_predict_single_fold,
                train_valid_iterator, ind, device_id) 
                for (ind, device_id) in zip(inds, device_ids))


            for elem in res:
                 models.extend(elem[0])
                 preds.extend(elem[1])
                 del elem

            self.models = models
            #THIS RELIES ON ASSUMPTION THAT PREDS ARE ORDERED CORRECTLY
            #WHICH THEY SEEM TO BE
            for n, (idx, _, _) in enumerate(train_valid_iterator):
                if isinstance(preds[n], (dask_cudf.DataFrame, dask_cudf.Series, dd.DataFrame, dd.Series)):
                    preds_arr[idx] += preds[n]\
                        .compute().values\
                        .reshape(preds[n].shape[0].compute(), -1)
                    counter_arr[idx] += 1
                else:
                    if isinstance(preds[n], np.ndarray):
                        preds[n] = cp.asarray(preds[n])
                    preds_arr[idx] += preds[n].reshape((preds[n].shape[0], -1))
                    counter_arr[idx] += 1
        else:
            # TODO: Make parallel version later
            for n, (idx, train, valid) in enumerate(train_valid_iterator):
                logger.info('\n===== Start working with fold {} for {} (orig) =====\n'.format(n, self._name))

                self.timer.set_control_point()
                model, pred = self.fit_predict_single_fold(train, valid)
                self.models.append(model)

                if isinstance(pred, (dask_cudf.DataFrame, dask_cudf.Series, dd.DataFrame, dd.Series)):

                    if idx is not None:
                        preds_arr[idx] += pred\
                            .compute().values\
                            .reshape(pred.shape[0].compute(), -1)
                        counter_arr[idx] += 1
                    else:
                        preds_arr += pred\
                            .compute().values\
                            .reshape(pred.shape[0].compute(), -1)
                        counter_arr += 1

                else:
                    if isinstance(pred, np.ndarray):
                        pred = cp.asarray(pred)
                    preds_arr[idx] += pred.reshape((pred.shape[0], -1))
                    counter_arr[idx] += 1

                self.timer.write_run_info()
                if (n + 1) != len(train_valid_iterator):
                    if self.timer.time_limit_exceeded():
                        logger.warning('Time limit exceeded after calculating fold {0}'\
                        .format(n))
                        break

            logger.debug('Time history {0}. Time left {1}'\
                .format(self.timer.get_run_results(), self.timer.time_left))

        '''if type(val_data) == DaskCudfDataset:
            preds_arr /= da.where(counter_arr == 0, 1, counter_arr)
            preds_arr = da.where(counter_arr == 0, cp.nan, preds_arr)

            cols = [self._name + "_" + str(x) for x in np.arange(self.n_classes)]
            preds_arr = dd.from_dask_array(preds_arr, columns=cols, meta=cudf.DataFrame(columns=cols)).persist()
            
        else:'''
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

        '''if type(dataset) == DaskCudfDataset:
            preds_ds = dataset.empty()
            preds_arr = None
        else:'''
        preds_ds = dataset.empty().to_cupy()
        preds_arr = None

        for model in self.models:

            pred = self.predict_single_fold(model, dataset)

            if isinstance(pred, (dask_cudf.DataFrame, dd.DataFrame)):
                pred = pred.compute().values
            elif isinstance(pred, cudf.DataFrame):
                pred = pred.values

            if preds_arr is None:
                preds_arr = pred
            else:
                preds_arr += pred

        '''if type(dataset) == DaskCudfDataset:
            preds_arr = preds_arr.to_dask_array(lengths=True).persist()'''

        preds_arr /= len(self.models)
        preds_arr = preds_arr.reshape((preds_arr.shape[0], -1))

        '''if type(dataset) == DaskCudfDataset:
            cols = [self._name + "_" + str(x) for x in np.arange(self.n_classes)]
            preds_arr = dd.from_dask_array(preds_arr, columns=cols, meta=cudf.DataFrame(columns=cols)).persist()'''

        preds_ds = self._set_prediction(preds_ds, preds_arr)

        return preds_ds
