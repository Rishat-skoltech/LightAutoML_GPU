#!/usr/bin/env python
# coding: utf-8

import logging
import time

import numpy as np
import pandas as pd
import cudf

from lightautoml.dataset.cp_cudf_dataset import CudfDataset
from lightautoml.dataset.roles import *
from lightautoml.dataset.utils import roles_parser
from lightautoml.tasks import Task

logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.DEBUG)

def test_gpu_datasets():
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
    cudf_data = cudf.DataFrame.from_pandas(data)

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
    logging.debug('Creating CudfDataset')
    cudf_dataset = CudfDataset(cudf_data, roles_parser(check_roles), task=task)

    start_time = time.time()
    numpy_dataset = cudf_dataset.to_numpy()
    print('numpy_dataset type is', type(numpy_dataset))
    logging.debug('Created numpy dataset. Time = {:.3f} sec'.format(time.time() - start_time))

    start_time = time.time()
    cupy_dataset = cudf_dataset.to_cupy()
    print('cupy_dataset type is', type(cupy_dataset))
    logging.debug('Created cupy dataset. Time = {:.3f} sec'.format(time.time() - start_time))

    start_time = time.time()
    numpy_dataset = cupy_dataset.to_numpy()
    pandas_dataset = cupy_dataset.to_pandas()
    print('numpy_dataset type is', type(numpy_dataset))
    print('pandas_dataset type is', type(pandas_dataset))
    logging.debug('Created numpy/pandas dataset. Time = {:.3f} sec'.format(time.time() - start_time))

if __name__ == '__main__':
    test_gpu_datasets()

