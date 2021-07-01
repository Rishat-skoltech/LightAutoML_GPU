#!/usr/bin/env python
# coding: utf-8

import logging
import os
import pickle
import time

#import cupy as cp
#import numpy as np

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
                       usecols=['AMT_CREDIT'])

    # Fix dates and convert to date type
    #print(data.head())
    #logging.debug('Fix dates and convert to date type')
    #data['BIRTH_DATE'] = np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))
    #data['EMP_DATE'] = np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
    #data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)
    #data = data["AMP_CREDIT"]

    # Create folds
    logging.debug('Create folds')
    data['fold'] = np.random.randint(0, 5, len(data))
    #data['TARGET'] = np.array(data['TARGET'])

    # Print data head
    logging.debug('Print data head')
    print(data.head())

    # # Set roles for columns
    logging.debug('Set roles for columns')
    check_roles = {
        NumericRole(np.float32): ['AMT_CREDIT', 'fold'],
    }

    # create Task
    task = Task('binary')
    # # Creating PandasDataSet
    logging.debug('Creating PandasDataset')
    start_time = time.time()
    pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)
    numpy_dataset = pd_dataset.to_numpy()
    logging.debug('Created numpy dataset. Time = {:.3f} sec'.format(time.time() - start_time))
    
    start_time = time.time()
    #_ = np.exp(numpy_dataset.data)
    logging.debug('SQRT value counted. Time = {:.3f} sec'.format(time.time() - start_time))
    
    start_time = time.time()
    cupy_dataset = numpy_dataset.to_cupy()
    logging.debug('Created cupy dataset. Time = {:.3f} sec'.format(time.time() - start_time))
    #print(type(cupy_dataset.data))
    #print(cupy_dataset.data.mean())

    
    start_time = time.time()
    numpy_dataset = cupy_dataset.to_numpy()
    pandas_dataset = cupy_dataset.to_pandas()
    logging.debug('Created numpy/pandas dataset. Time = {:.3f} sec'.format(time.time() - start_time))
    #print(type(numpy_dataset.data))