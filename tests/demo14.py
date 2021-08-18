#!/usr/bin/env python
# coding: utf-8

import logging
import time

from lightautoml.dataset.cp_cudf_dataset import CudfDataset
from lightautoml.dataset.daskcudf_dataset import DaskCudfDataset
from lightautoml.dataset.roles import *
from lightautoml.dataset.utils import roles_parser
from lightautoml.tasks import Task
import numpy as np
import pandas as pd
import cudf
import dask_cudf

logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.DEBUG)

def test_gpu_datasets_more():
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


    cudf_data = cudf.DataFrame.from_pandas(data)
    daskcudf_data = dask_cudf.from_cudf(cudf_data, npartitions=4)

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
    cudf_dataset = CudfDataset(cudf_data, roles_parser(check_roles), task=task)
    daskcudf_dataset = DaskCudfDataset(daskcudf_data, roles_parser(check_roles), task=task)

    logging.debug('CudfDataset created. Time = {:.3f} sec'.format(time.time() - start_time))
    # # Print cudf dataset feature roles
    logging.debug('Print cudf dataset feature roles')

    roles = cudf_dataset.roles
    for role in roles:
        logging.debug('{}: {}'.format(role, roles[role]))
    # # Feature selection part
    logging.debug('Feature selection part')

    # Testing iloc functionality
    logging.debug('Testing slices functionality')
    slice1 = cudf_dataset._get_cols(cudf_dataset.data, 2)
    slice2 = cudf_dataset._get_rows(cudf_dataset.data, 1)
    #slice3 = cudf_dataset._get_2d(cudf_dataset, (2,3))
    print('Slice 1:', slice1)
    print('Slice 2:', slice2)
    logging.debug('Testing DaskCudfDataset iloc')
    row1 = daskcudf_dataset[50]
    row2 = daskcudf_dataset[450]
    print('row 1:', row1.data.compute())
    print('row 2:', row2.data.compute())
    return 0

if __name__ == '__main__':
    test_gpu_datasets_more()
