"""Wrapped LightGBM for tabular datasets."""

import logging
from copy import copy, deepcopy
from typing import Optional, Callable, Tuple, Dict

import xgboost as xgb
from xgboost import dask as dxgb

import numpy as np
import pandas as pd

from optuna.trial import Trial
from pandas import Series

from ..dataset.daskcudf_dataset import DaskCudfDataset
from .base_gpu import TabularMLAlgo_gpu, TabularDatasetGpu
from .tuning.optuna import OptunaTunableMixin
from ..pipelines.selection.base import ImportanceEstimator
from ..utils.logging import get_logger
from ..validation.base import TrainValidIterator

logger = get_logger(__name__)

class BoostXGB(OptunaTunableMixin, TabularMLAlgo_gpu, ImportanceEstimator):
    """Gradient boosting on decision trees from LightGBM library.

    default_params: All available parameters listed in lightgbm documentation:

        - https://lightgbm.readthedocs.io/en/latest/Parameters.html

    freeze_defaults:

        - ``True`` :  params may be rewritten depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """
    _parallel_folds: bool = False
    _name: str = 'XGB'

    _default_params = {
        'tree_method':'gpu_hist',
        'predictor':'gpu_predictor',
        'task': 'train',
        "learning_rate": 0.05,
        "max_leaves": 128,
        "max_depth": 0,
        "verbosity": 0,
        "reg_alpha": 1,
        "reg_lambda": 0.0,
        "gamma": 0.0,
        'max_bin': 255,
        'n_estimators': 3000,
        'early_stopping_rounds': 100,
        'random_state': 42
    }

    def _infer_params(self) -> Tuple[dict, int, int, int, Optional[Callable], Optional[Callable]]:
        """Infer all parameters in lightgbm format.

        Returns:
            Tuple (params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval).
            About parameters: https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/engine.html

        """
        # TODO: Check how it works with custom tasks
        params = copy(self.params)
        early_stopping_rounds = params.pop('early_stopping_rounds')
        num_trees = params.pop('n_estimators')

        root_logger = logging.getLogger()
        level = root_logger.getEffectiveLevel()

        if level in (logging.CRITICAL, logging.ERROR, logging.WARNING):
            verbose_eval = False
        elif level == logging.INFO:
            verbose_eval = 100
        else:
            verbose_eval = 10

        # get objective params
        loss = self.task.losses['xgb_gpu']
        params['objective'] = loss.fobj_name
        fobj = loss.fobj

        # get metric params
        params['metric'] = loss.metric_name
        feval = loss.feval

        params['num_class'] = self.n_classes
        # add loss and tasks params if defined
        params = {**params, **loss.fobj_params, **loss.metric_params}

        return params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """

        # TODO: use features_num
        # features_num = len(train_valid_iterator.features())

        #will this work for two partitions?
        rows_num = len(train_valid_iterator.train)

        task = train_valid_iterator.train.task.name

        suggested_params = copy(self.default_params)

        if self.freeze_defaults:
            # if user change defaults manually - keep it
            return suggested_params

        if task == 'reg':
            suggested_params = {
                "learning_rate": 0.05,
                "max_leaves": 32
            }

        if rows_num <= 10000:
            init_lr = 0.01
            ntrees = 3000
            es = 200

        elif rows_num <= 20000:
            init_lr = 0.02
            ntrees = 3000
            es = 200

        elif rows_num <= 100000:
            init_lr = 0.03
            ntrees = 1200
            es = 200
        elif rows_num <= 300000:
            init_lr = 0.04
            ntrees = 2000
            es = 100
        else:
            init_lr = 0.05
            ntrees = 2000
            es = 100

        if rows_num > 300000:
            suggested_params['max_leaves'] = 128 if task == 'reg' else 244
        elif rows_num > 100000:
            suggested_params['max_leaves'] = 64 if task == 'reg' else 128
        elif rows_num > 50000:
            suggested_params['max_leaves'] = 32 if task == 'reg' else 64
            # params['reg_alpha'] = 1 if task == 'reg' else 0.5
        elif rows_num > 20000:
            suggested_params['max_leaves'] = 32 if task == 'reg' else 32
            suggested_params['reg_alpha'] = 0.5 if task == 'reg' else 0.0
        elif rows_num > 10000:
            suggested_params['max_leaves'] = 32 if task == 'reg' else 64
            suggested_params['reg_alpha'] = 0.5 if task == 'reg' else 0.2
        elif rows_num > 5000:
            suggested_params['max_leaves'] = 24 if task == 'reg' else 32
            suggested_params['reg_alpha'] = 0.5 if task == 'reg' else 0.5
        else:
            suggested_params['max_leaves'] = 16 if task == 'reg' else 16
            suggested_params['reg_alpha'] = 1 if task == 'reg' else 1

        suggested_params['learning_rate'] = init_lr
        suggested_params['n_estimators'] = ntrees
        suggested_params['early_stopping_rounds'] = es

        return suggested_params

    def sample_params_values(self, trial: Trial, suggested_params: Dict, estimated_n_trials: int) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        logger.debug('Suggested parameters:')
        logger.debug(suggested_params)

        trial_values = copy(suggested_params)

        trial_values['max_depth'] = trial.suggest_int(
            name='max_depth',
            low=3,
            high=7
        )

        trial_values['max_leaves'] = trial.suggest_int(
            name='max_leaves',
            low=16,
            high=255,
        )

        if estimated_n_trials > 30:
            trial_values['min_child_weight'] = trial.suggest_loguniform(
                name='min_child_weight',
                low=1e-3,
                high=10.0,
            )

        if estimated_n_trials > 100:
            trial_values['reg_alpha'] = trial.suggest_loguniform(
                name='reg_alpha',
                low=1e-8,
                high=10.0,
            )
            trial_values['reg_lambda'] = trial.suggest_loguniform(
                name='reg_lambda',
                low=1e-8,
                high=10.0,
            )

        return trial_values

    def fit_predict_single_fold(self, train: TabularDatasetGpu, valid: TabularDatasetGpu, part_id: int = None, dev_id: int = 0) -> Tuple[xgb.Booster, np.ndarray]:
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values)

        """
        train_target = train.target
        train_weights = train.weights
        valid_target = valid.target
        valid_weights = valid.weights
        train_data = train.data
        valid_data = valid.data
        if type(train) == DaskCudfDataset:
            assert part_id is not None, 'fit_predict_single_fold: partition id should be set if data is distributed'
            train_target = train_target.compute()
            #train_target = train_target.get_partition(part_id).compute()
            if train_weights is not None:
                train_weights = train_weights.compute()
                #train_weights = train_weights.get_partition(part_id).compute()
            valid_target = valid_target.compute()
            #valid_target = valid_target.get_partition(part_id).compute()
            if valid_weights is not None:
                #valid_weights = valid_weights.get_partition(part_id).compute()
                valid_weights = valid_weights.compute()
            train_data = train_data.compute()
            #train_data = train_data.get_partition(part_id).compute()
            valid_data = valid_data.compute()
            #valid_data = valid_data.get_partition(part_id).compute()
        params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval = self._infer_params()
        train_target, train_weight = self.task.losses['xgb_gpu'].fw_func(train_target, train_weights)
        valid_target, valid_weight = self.task.losses['xgb_gpu'].fw_func(valid_target, valid_weights)
        xgb_train = xgb.DMatrix(train_data, label=train_target, weight=train_weight)
        xgb_valid = xgb.DMatrix(valid_data, label=valid_target, weight=valid_weight)
        model = xgb.train(params, xgb_train, num_boost_round=num_trees, evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
                          obj=fobj, feval=feval, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval
                          )
        val_pred = model.inplace_predict(valid_data)
        val_pred = self.task.losses['xgb_gpu'].bw_func(val_pred)
        return model, val_pred

    def predict_single_fold(self, model: xgb.Booster, dataset: TabularDatasetGpu, part_id: int = None) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Lightgbm object.
            dataset: Test Dataset.

        Return:
            Predicted target values.

        """
        dataset_data = dataset.data
        if type(dataset) == DaskCudfDataset:
            assert part_id is not None, 'predict_single_fold: partition id should be set if data is distributed'
            dataset_data = dataset_data.compute()
            #dataset_data = dataset_data.get_partition(part_id).compute()

        pred = self.task.losses['xgb_gpu'].bw_func(model.inplace_predict(dataset_data))

        return pred

    def get_features_score(self) -> pd.Series:
        """Computes feature importance as mean values of feature importance provided by lightgbm per all models.

        Returns:
            Series with feature importances.

        """

        #FIRST SORT TO FEATURES AND THEN SORT BACK TO IMPORTANCES - BAD
        imp = 0
        for model in self.models:
            val = model.get_score(importance_type='gain')
            sorted_list = [0.0 if val.get(i) is None else val.get(i) for i in self.features]
            scores = np.array(sorted_list)
            imp = imp + scores

        imp = imp / len(self.models)
        
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

    def fit(self, train_valid: TrainValidIterator):
        """Just to be compatible with :class:`~lightautoml.pipelines.selection.base.ImportanceEstimator`.

        Args:
            train_valid: Classic cv-iterator.

        """
        self.fit_predict(train_valid)
        
class BoostXGB_dask(BoostXGB):

    _parallel_folds: bool = False

    def __init__(self, client, *args, **kwargs):

        self.client = client
        super().__init__(*args, **kwargs)
        
        
    #CHECK IF EVERYTHING IS CORRECT
    #BECAUSE: "Copying TaskTime may affect the parent PipelineTime,
    #so copy will create new unlimited TaskTimer
    def __deepcopy__(self, memo):

        new_inst = type(self).__new__(self.__class__)
        new_inst.client = self.client
        
        for k,v in super().__dict__.items():
            if k != 'client':
                setattr(new_inst, k, deepcopy(v, memo))
        return new_inst
        
    def fit_predict_single_fold(self, train: DaskCudfDataset, valid: DaskCudfDataset, part_id: int = None, dev_id: int = 0) -> Tuple[dxgb.Booster, np.ndarray]:
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values)

        """

        params, num_trees, early_stopping_rounds, verbose_eval, fobj, feval = self._infer_params()

        train_target, train_weight = self.task.losses['xgb_mgpu'].fw_func(train.target, train.weights)
        valid_target, valid_weight = self.task.losses['xgb_mgpu'].fw_func(valid.target, valid.weights)

        xgb_train = dxgb.DaskDeviceQuantileDMatrix(self.client, train.data, label=train_target, weight=train_weight)
        xgb_valid = dxgb.DaskDeviceQuantileDMatrix(self.client, valid.data, label=valid_target, weight=valid_weight)
        model = dxgb.train(self.client, params, xgb_train, num_boost_round=num_trees, evals=[(xgb_train, 'train'), (xgb_valid, 'valid')],
                          obj=fobj, feval=feval, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose_eval
                          )

        val_pred = dxgb.inplace_predict(self.client, model, valid.data)
        val_pred = self.task.losses['xgb_mgpu'].bw_func(val_pred)

        return model, val_pred

    def predict_single_fold(self, model: dxgb.Booster, dataset: TabularDatasetGpu) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Lightgbm object.
            dataset: Test Dataset.

        Return:
            Predicted target values.

        """
        pred = self.task.losses['xgb_mgpu'].bw_func(dxgb.inplace_predict(self.client, model, dataset.data))

        return pred

    def get_features_score(self) -> pd.Series:
        """Computes feature importance as mean values of feature importance provided by lightgbm per all models.

        Returns:
            Series with feature importances.

        """

        #FIRST SORT TO FEATURES AND THEN SORT BACK TO IMPORTANCES - BAD
        imp = 0
        for model in self.models:
            val = model['booster'].get_score(importance_type='gain')
            sorted_list = [0.0 if val.get(i) is None else val.get(i) for i in self.features]
            scores = np.array(sorted_list)
            imp = imp + scores

        imp = imp / len(self.models)
        
        return pd.Series(imp, index=self.features).sort_values(ascending=False)

