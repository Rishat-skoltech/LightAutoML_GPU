#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pickle
import time

from lightautoml.dataset.np_pd_dataset_cupy import *
from lightautoml.dataset.roles import *
from lightautoml.dataset.utils import roles_parser
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.tuning.optuna import OptunaTuner
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.tasks import Task
from lightautoml.validation.np_iterators import FoldsIterator

logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.DEBUG)

def test_manual_pipeline():
    # Read data from file
    logging.debug('Read data from file')
    data = pd.read_csv('../example_data/test_data_files/sampled_app_train.csv',
                       usecols=['TARGET', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT',
                                'NAME_TYPE_SUITE', 'AMT_GOODS_PRICE',
                                'DAYS_BIRTH', 'DAYS_EMPLOYED'])

    # Fix dates and convert to date type
    logging.debug('Fix dates and convert to date type')
    data['BIRTH_DATE'] = np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))
    data['EMP_DATE'] = np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
    data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

    # Create folds
    logging.debug('Create folds')
    data['__fold__'] = np.random.randint(0, 5, len(data))

    # Print data head
    logging.debug('Print data head')
    print(data.head())

    # # Set roles for columns
    logging.debug('Set roles for columns')
    check_roles = {
        TargetRole(): 'TARGET',
        CategoryRole(dtype=str): ['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE'],
        NumericRole(np.float32): ['AMT_CREDIT', 'AMT_GOODS_PRICE'],
        DatetimeRole(seasonality=['y', 'm', 'wd']): ['BIRTH_DATE', 'EMP_DATE'],
        FoldsRole(): '__fold__'
    }

    # create Task
    task = Task('binary')
    # # Creating PandasDataSet
    logging.debug('Creating CudfDataset from PandasDataset')
    start_time = time.time()
    pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)
    cudf_dataset = pd_dataset.to_cudf()

    logging.debug('CudfDataset created. Time = {:.3f} sec'.format(time.time() - start_time))
    # # Print pandas dataset feature roles
    logging.debug('Print cudf dataset feature roles')

    roles = cudf_dataset.roles
    for role in roles:
        logging.debug('{}: {}'.format(role, roles[role]))
    # # Feature selection part
    logging.debug('Feature selection part')

    #############TO BE DONE###########################
    exit()
test_manual_pipeline()