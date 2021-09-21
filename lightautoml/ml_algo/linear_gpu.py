"""Linear models for tabular datasets."""

from copy import copy, deepcopy
from typing import Tuple, Union, Sequence

import cupy as cp

from cuml.linear_model import LogisticRegression, ElasticNet, Lasso

from .base_gpu import TabularMLAlgo, TabularDataset
from .torch_based.linear_model_cupy import TorchBasedLinearEstimator, TorchBasedLinearRegression, \
    TorchBasedLogisticRegression

from .torch_based.linear_model_distributed import TorchBasedLinearEstimator as TLE_dask, \
    TorchBasedLogisticRegression as TLR_dask, TorchBasedLinearRegression as TLinR_dask
from ..dataset.np_pd_dataset_cupy import CudfDataset, DaskCudfDataset, CupyDataset
from ..utils.logging import get_logger
from ..validation.base import TrainValidIterator

logger = get_logger(__name__)

LinearEstimator = Union[LogisticRegression, ElasticNet, Lasso]


class LinearLBFGS(TabularMLAlgo):
    """LBFGS L2 regression based on torch.


    default_params:

        - cs: List of regularization coefficients.
        - max_iter: Maximum iterations of L-BFGS.
        - tol: The tolerance for the stopping criteria.
        - early_stopping: Maximum rounds without improving.

    freeze_defaults:

        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """
    _name: str = 'LinearL2'

    _default_params = {

        'tol': 1e-6,
        'max_iter': 100,
        'cs': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10,
               50, 100, 500, 1000, 5000, 10000, 50000, 100000],
        'early_stopping': 2

    }

    def _infer_params(self) -> TorchBasedLinearEstimator:

        params = copy(self.params)
        params['loss'] = self.task.losses['torch'].loss
        params['metric'] = self.task.losses['torch'].metric_func

        if self.task.name in ['binary', 'multiclass']:
            model = TorchBasedLogisticRegression(output_size=self.n_classes, **params)
        elif self.task.name == 'reg':

            model = TorchBasedLinearRegression(output_size=1, **params)
        else:
            raise ValueError('Task not supported')

        return model
    def _infer_params_dask(self) -> TorchBasedLinearEstimator:

        params = copy(self.params)
        params['loss'] = self.task.losses['torch'].loss
        params['metric'] = self.task.losses['torch'].metric_func

        if self.task.name in ['binary', 'multiclass']:
            model = TLR_dask(output_size=self.n_classes, **params)
        elif self.task.name == 'reg':

            model = TLinR_dask(output_size=1, **params)
        else:
            raise ValueError('Task not supported')

        return model

    # TODO: only for debugging purposes!

    def fit_predict_dask_(self, train_valid_iterator: TrainValidIterator):
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
        outp_dim = 1
        if self.task.name == 'multiclass':
            outp_dim = int(train_valid_iterator.train.target.compute().values.max()+1)
        # save n_classes to infer params
        self.n_classes = outp_dim



        # for n, (idx, train, valid) in enumerate(train_valid_iterator):
        model = self._infer_params_dask()
        print(model.__class__.__name__)
        print(model.model)
        # print(train.data.shape)
        print(type(train_valid_iterator.train))
        model.fit(train_valid_iterator.train.data, train_valid_iterator.train.target)

        # print(f'round {n}')

        return "Passed"


    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:

        suggested_params = copy(self.default_params)
        train = train_valid_iterator.train
        suggested_params['categorical_idx'] = [n for (n, x) in enumerate(train.features) if train.roles[x].name == 'Category']

        suggested_params['embed_sizes'] = ()
        if len(suggested_params['categorical_idx']) > 0:
            suggested_params['embed_sizes'] = train.data[:, suggested_params['categorical_idx']].max(axis=0).astype(cp.int32) + 1

        suggested_params['data_size'] = train.shape[1]

        return suggested_params


    def fit_predict_single_fold(self, train: TabularDataset, valid: TabularDataset
                                ) -> Tuple[TorchBasedLinearEstimator, cp.ndarray]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        # TODO: make revision upon finish of dask+torch
        if type(train) is CupyDataset or type(train) is CudfDataset:
            model = self._infer_params()

            if type(train) is CudfDataset:
                train = train.to_cupy()
                valid = valid.to_cupy()

        if type(train) is DaskCudfDataset:
            model = self._infer_params_dask()

        model.fit(train.data, train.target, train.weights, valid.data, valid.target, valid.weights)
        val_pred = model.predict(valid.data)
        return model, val_pred

    def predict_single_fold(self, model: TorchBasedLinearEstimator, dataset: TabularDataset) -> cp.ndarray:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: ``CupyDataset`` used for prediction.

        Returns:
            Predictions for input dataset.

        """
        pred = model.predict(dataset.data)

        return pred


class LinearL1CD(TabularMLAlgo):
    """Coordinate descent based on cuml implementation."""
    _name: str = 'LinearElasticNet'

    _default_params = {

        'tol': 1e-3,
        'max_iter': 100,
        'cs': [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000, 100000, 1000000],
        'early_stopping': 2,
        'l1_ratios': (1,),
        'solver': 'qn'

    }

    def _infer_params(self) -> Tuple[LinearEstimator, Sequence[float], Sequence[float], int]:

        params = copy(self.params)
        l1_ratios = params.pop('l1_ratios')
        early_stopping = params.pop('early_stopping')
        cs = params.pop('cs')

        if self.task.name in ['binary', 'multiclass']:

            if l1_ratios == (1,):
                model = LogisticRegression(penalty='l1', **params)
            else:
                model = LogisticRegression(penalty='elasticnet', **params)

        elif self.task.name == 'reg':
            params.pop('solver')
            if l1_ratios == (1,):
                model = Lasso(**params)
            else:
                model = ElasticNet(**params)

        else:
            raise AttributeError('Task not supported')

        return model, cs, l1_ratios, early_stopping

    def init_params_on_input(self, train_valid_iterator: TrainValidIterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """

        suggested_params = copy(self.default_params)
        task = train_valid_iterator.train.task
        assert 'cuml' in task.losses, 'Cuml loss should be defined'

        if task.name == 'reg':
            # suggested_params['cs'] = list(map(lambda x: 1 / (2 * x), suggested_params['cs']))
            suggested_params['cs'] = [1 / (2 * i) for i in suggested_params['cs']]

        return suggested_params

    def _predict_w_model_type(self, model, data):

        if self.task.name == 'binary':
            pred = model.predict_proba(data)[:, 1]

        elif self.task.name == 'reg':
            pred = model.predict(data)

        elif self.task.name == 'multiclass':
            pred = model.predict_proba(data)

        else:
            raise ValueError('Task not suppoted')

        return pred

    def fit_predict_single_fold(self, train: TabularDataset, valid: TabularDataset) -> Tuple[LinearEstimator, cp.ndarray]:
        """Train on train dataset and predict on holdout dataset.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Target predictions for valid dataset.

        """
        if type(train) is CudfDataset:
            train = train.to_cupy()
            valid = valid.to_cupy()

        _model, cs, l1_ratios, early_stopping = self._infer_params()

        train_target, train_weight = self.task.losses['cuml'].fw_func(train.target, train.weights)
        valid_target, valid_weight = self.task.losses['cuml'].fw_func(valid.target, valid.weights)

        model = deepcopy(_model)

        best_score = -cp.inf
        best_pred = None
        best_model = None

        metric = self.task.losses['cuml'].metric_func

        for l1_ratio in sorted(l1_ratios, reverse=True):

            try:
                model.set_params(**{'l1_ratio': l1_ratio})
            except ValueError:
                pass

            model = deepcopy(_model)

            c_best_score = -cp.inf
            c_best_pred = None
            c_best_model = None
            es = 0

            for n, c in enumerate(cs):

                try:
                    model.set_params(**{'C': c})
                except ValueError:
                    model.set_params(**{'alpha': c})

                model.fit(train.data, train_target, train_weight)

                if cp.allclose(model.coef_, 0):
                    if n == (len(cs) - 1):
                        logger.warning('All model coefs are 0. Model with l1_ratio {0} is dummy'.format(l1_ratio), UserWarning)
                    else:
                        logger.debug('C = {0} all model coefs are 0'.format(c))
                        continue

                pred = self._predict_w_model_type(model, valid.data)
                score = metric(valid_target, pred, valid_weight)

                logger.debug('C = {0}, l1_ratio = {1}, score = {2}'.format(c, 1, score))

                # TODO: check about greater and equal
                if score >= c_best_score:
                    c_best_score = score
                    c_best_pred = deepcopy(pred)
                    es = 0
                    c_best_model = deepcopy(model)
                else:
                    es += 1

                if es >= early_stopping:
                    logger.debug('Early stopping..')
                    break

                if self.timer.time_limit_exceeded():
                    logger.info('Time limit exceeded')
                    break

                # TODO: Think about is it ok to check time inside train loop?
                if (model.coef_ != 0).all():
                    logger.debug('All coefs are nonzero')
                    break

            if c_best_score >= best_score:
                best_score = c_best_score
                best_pred = deepcopy(c_best_pred)
                best_model = deepcopy(c_best_model)

            if self.timer.time_limit_exceeded():
                logger.info('Time limit exceeded')
                break

        val_pred = self.task.losses['cuml'].bw_func(best_pred)

        return best_model, val_pred

    def predict_single_fold(self, model: LinearEstimator, dataset: TabularDataset) -> cp.ndarray:
        """Implements prediction on single fold.

        Args:
            model: Model uses to predict.
            dataset: Dataset used for prediction.

        Returns:
            Predictions for input dataset.

        """

        pred = self.task.losses['cuml'].bw_func(self._predict_w_model_type(model, dataset.data))

        return pred
