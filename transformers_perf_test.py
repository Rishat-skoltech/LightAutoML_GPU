import os
import pickle
import time

import numpy as np

from lightautoml.dataset.np_pd_dataset_cupy import *
from lightautoml.dataset.roles import *
from lightautoml.dataset.utils import roles_parser

from lightautoml.tasks import Task

from lightautoml.transformers import numeric_gpu, categorical_gpu, datetime_gpu
from lightautoml.transformers import numeric, categorical, datetime as datetime_cpu

from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

numeric_transformers_cpu = (numeric.FillnaMedian(),
                            numeric.StandardScaler(),
                            numeric.NaNFlags(),
                            numeric.FillInf(),
                            numeric.QuantileBinning())

numeric_transformers_gpu = (numeric_gpu.FillnaMedian(),
                            numeric_gpu.StandardScaler(),
                            numeric_gpu.NaNFlags(),
                            numeric_gpu.FillInf(),
                            numeric_gpu.QuantileBinning())


def load_dataset(path='./example_data/test_data_files/sampled_app_train.csv'):
    data = pd.read_csv(path,
                       usecols=['TARGET', 'NAME_CONTRACT_TYPE', 'AMT_CREDIT',
                                'NAME_TYPE_SUITE', 'AMT_GOODS_PRICE',
                                'DAYS_BIRTH', 'DAYS_EMPLOYED'])

    # Fix dates and convert to date type

    data['BIRTH_DATE'] = np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))
    data['EMP_DATE'] = np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
    data.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, inplace=True)

    # Create folds
    data['__fold__'] = np.random.randint(0, 5, len(data))

    # # Set roles for columns

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
    pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)

    cudf_dataset = pd_dataset.to_cudf()

    check_roles = {
        CategoryRole(dtype=str): ['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE'],
    }
    categorical_dataset = CudfDataset(data=cudf_dataset.data[['NAME_CONTRACT_TYPE', \
                                                              'NAME_TYPE_SUITE']].copy(),
                                      roles=roles_parser(check_roles),
                                      task=Task('binary'))


    return cudf_dataset


def test_features(num_dataset, algo):
    start = datetime.now()

    fitted = algo.fit_transform(dataset=num_dataset)
    end = datetime.now()
    time_perf = (end - start).total_seconds()
    return time_perf, fitted


def create_numeric_dataset(col_size=10, rows=10000, with_nans=False):
    data = np.random.randn(rows, col_size)
    if with_nans:
        data.ravel()[np.random.choice(data.size, int(0.1 * rows) * col_size, replace=False)] = np.nan
    features = ['feat_' + str(_) for _ in range(col_size)]
    check_roles_ = {
        NumericRole(np.float32): features,
    }
    numeric_dataset_ = CupyDataset(data=data,
                                   features=features,
                                   roles=roles_parser(check_roles_),
                                   task=Task('binary')).to_cudf()

    return numeric_dataset_


def test_numeric_datasets(col_size=10):
    time_cpu = {}
    time_gpu = {}

    for i, (algo_cpu, algo_gpu) in enumerate(zip(numeric_transformers_cpu,
                                                 numeric_transformers_gpu)):
        time_cpu[algo_cpu.__class__.__name__] = {}
        time_gpu[algo_gpu.__class__.__name__] = {}

        for size in (10, 20, 30, 50, 70, 100, 200, 500, 1000, 2000, 5000, 10000):
            size *= 1000
            print("Current size:", size)
            if i == 0:
                cudf_dataset = create_numeric_dataset(col_size=col_size, rows=size, with_nans=True)
            else:
                cudf_dataset = create_numeric_dataset(col_size=col_size, rows=size)

            time_cpu[algo_cpu.__class__.__name__][size], _ = test_features(cudf_dataset.to_pandas(), algo_cpu)
            time_gpu[algo_gpu.__class__.__name__][size], _ = test_features(cudf_dataset, algo_gpu)

    return time_cpu, time_gpu


def test_categorical_datasets():
    pass


def build_plots(time_cpu, time_gpu):
    for algo_cpu, algo_gpu in zip(numeric_transformers_cpu,
                                  numeric_transformers_gpu):
        plt.figure(figsize=(12, 6))
        plt.title(algo_cpu.__class__.__name__, fontsize=24)
        plt.grid()
        plt.axes().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("data size (rows)", fontsize=18)
        plt.ylabel("perf time (seconds)", fontsize=18)
        vals_cpu = sorted(time_cpu[algo_cpu.__class__.__name__].items())
        x, y = zip(*vals_cpu)
        plt.plot(x, y, label='cpu')
        vals_gpu = sorted(time_gpu[algo_gpu.__class__.__name__].items())
        x, y = zip(*vals_gpu)
        plt.plot(x, y, label='gpu')
        plt.legend()
        plt.savefig('./results_test/' + algo_cpu.__class__.__name__ + '.png')


time_cpu, time_gpu = test_numeric_datasets()
build_plots(time_cpu, time_gpu)
