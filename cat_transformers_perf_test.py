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

cat_transformers_cpu = (categorical.CatIntersectstions(),
                        categorical.LabelEncoder(),
                        categorical.OHEEncoder(make_sparse=True),
                        categorical.FreqEncoder(),
                        categorical.TargetEncoder(),
                        )

cat_transformers_gpu = (categorical_gpu.CatIntersectstions(),
                        categorical_gpu.LabelEncoder(),
                        categorical_gpu.OHEEncoder(make_sparse=True),
                        categorical_gpu.FreqEncoder(),
                        categorical_gpu.TargetEncoder(),
                        )

target = None
fold = None


def load_dataset(path='./example_data/test_data_files/sampled_app_train.csv', nrows=10000):
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
    global target
    target = cudf.Series(np.random.randint(0, 1, nrows))
    global fold
    fold = cudf.Series(np.random.randint(0,11, nrows))

    # # Creating PandasDataSet
    pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)

    cudf_dataset = pd_dataset.to_cudf()

    check_roles = {
        CategoryRole(dtype=str): ['NAME_CONTRACT_TYPE', 'NAME_TYPE_SUITE'],
    }

    data = cudf_dataset.data[['NAME_CONTRACT_TYPE',
                              'NAME_TYPE_SUITE']].copy()
    data = data.sample(nrows, replace=True).reset_index().drop(['index'], axis=1)
    categorical_dataset = CudfDataset(data=data,
                                      roles=roles_parser(check_roles),
                                      task=Task('binary'))
    categorical_dataset.target = target
    return categorical_dataset


def test_features(num_dataset, algo):
    start = datetime.now()

    fitted = algo.fit_transform(dataset=num_dataset)
    end = datetime.now()
    time_perf = (end - start).total_seconds()
    return time_perf, fitted


def test_cat_datasets():
    time_cpu = {}
    time_gpu = {}

    for size in (10, 20, 30, 50, 70, 100, 200, 500, 1000, 2000, 5000, 10000):
        size *= 1000

        for i, (algo_cpu, algo_gpu) in enumerate(zip(cat_transformers_cpu,
                                                     cat_transformers_gpu)):
            print(algo_cpu.__class__.__name__, '...')
            if time_cpu.get(algo_cpu.__class__.__name__) is None:
                time_cpu[algo_cpu.__class__.__name__] = {}
            if time_gpu.get(algo_cpu.__class__.__name__) is None:
                time_gpu[algo_gpu.__class__.__name__] = {}

            if i < 2:
                cat_dataset = load_dataset(nrows=size)

                time_cpu[algo_cpu.__class__.__name__][size], encoded_dataset_cpu = test_features(cat_dataset.to_pandas(),
                                                                                                 algo_cpu)
                time_gpu[algo_gpu.__class__.__name__][size], encoded_dataset = test_features(cat_dataset,
                                                                                             algo_gpu)
            else:
                encoded_dataset_cpu.target = target.to_pandas()
                encoded_dataset.target = target

                encoded_dataset_cpu.folds = fold.to_pandas().values
                encoded_dataset.folds = fold

                print(encoded_dataset_cpu.folds.shape)
                time_cpu[algo_cpu.__class__.__name__][size], _ = test_features(encoded_dataset_cpu,
                                                                               algo_cpu)
                time_gpu[algo_gpu.__class__.__name__][size], _ = test_features(encoded_dataset,
                                                                               algo_gpu)
    return time_cpu, time_gpu



def build_plots(time_cpu, time_gpu):
    for algo_cpu, algo_gpu in zip(cat_transformers_cpu,
                                  cat_transformers_gpu):
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


time_cpu, time_gpu = test_cat_datasets()
build_plots(time_cpu, time_gpu)
