"""Linear models based on Torch library."""

import numpy as np
import cupy as cp

from time import perf_counter

import numpy as np
import pandas as pd

from lightautoml.tasks import Task
from sklearn.model_selection import train_test_split

import dask_cudf
import cudf

from lightautoml.ml_algo.torch_based.linear_model_cupy import TorchBasedLogisticRegression as TorchBasedLogisticRegression_gpu
from lightautoml.ml_algo.torch_based.linear_model import TorchBasedLogisticRegression

import random

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from numba import jit
import string


RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))

@jit(nopython=True)
def gen_cols(n_cols):
    cols = [""]*n_cols
    for i in range(n_cols):
        cols[i] = "col_" + str(i)
    return cols

def gen_string_data(n, n_str):
    string_db = ["algorithm", "analog", "app", "application", "array",
                 "backup", "bandwidth", "binary", "bit", "byte"]#,
                 #"bitmap", "blog", "bookmark", "boot", "broadband",
                 #"browser" , "buffer", "bug"]
    inds = np.random.randint(0, len(string_db), (n, n_str))
    output = np.empty(inds.shape, dtype=object)
    for i in range(inds.shape[0]):
        for j in range(inds.shape[1]):
            output[i][j] = string_db[inds[i][j]]

    return output

def generate_data(n, n_num, n_cat, n_date, n_str, max_n_cat):
    print("Generating dummy data")
    n_cols = n_num+n_cat+n_str+n_date
    cols = gen_cols(n_cols)
    data = np.random.random((n, n_num))*100-50

    category_data = np.random.randint(0, np.random.randint(1,max_n_cat), (n, n_cat))
    string_data = gen_string_data(n, n_str)

    string_data = np.reshape(string_data, (n, n_str))

    date_data = np.random.randint(0, 1000,
                               (n, n_date)).astype(np.dtype("timedelta64[D]")) \
                               + np.datetime64("2018-01-01")

    data = pd.DataFrame(data, columns = cols[:n_num]).astype('f')
    
    ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
    for row, col in random.sample(ix, int(round(.0*len(ix)))):
        data.iat[row, col] = np.nan
    
    nn = len(data.columns)
    for i in range(n_cat):
        data[cols[nn+i]] = pd.Series(category_data[:,i]).astype('f')
    nn = len(data.columns)
    for i in range(n_str):
        data[cols[nn+i]] = pd.Series(string_data[:,i]).astype(object)
    nn = len(data.columns)
    for i in range(n_date):
        data[cols[nn+i]] = pd.Series(date_data[:,i])
        
    data['target'] = pd.Series(np.random.randint(0, 4, n)).astype('i')

    print("Shape of the dummy data:", data.shape)
    print("Size of the dummy data:",
          round(data.memory_usage(deep=True).sum()/1024./1024.,4), "MB.")
    return 'target', cols, data


if __name__ == "__main__":
    task = Task('multiclass', metric = 'crossentropy', device='gpu') 

    target, _, data = generate_data(n=40, n_num=3, n_cat=2, n_date=0,
                                    n_str=0, max_n_cat=10)
    print(data)

    params = {

        'tol': 1e-6,
        'max_iter': 100,
        'cs': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10,
               50, 100, 500, 1000, 5000, 10000, 50000, 100000],
        'early_stopping': 2

    }

    cols = ['col_0', 'col_1', 'col_2']
    weights = 'col_3'

    params['loss'] = task.losses['torch'].loss
    params['metric'] = task.losses['torch'].metric_func


    train_data, test_data = train_test_split(data, 
                                         test_size=20, 
                                         stratify=data[target], 
                                         random_state=42)
    
    cudf_data = cudf.DataFrame.from_pandas(data, nan_as_null=False)
    
    train_cudf = cudf.DataFrame.from_pandas(train_data, nan_as_null=False)
    test_cudf = cudf.DataFrame.from_pandas(test_data, nan_as_null=False)
    
    daskcudf_data = dask_cudf.from_cudf(cudf_data, npartitions=1)
    
    model = TorchBasedLogisticRegression_gpu(data_size = 3, output_size = 4, **params)
    model.fit(train_cudf[cols], train_cudf[target], train_cudf[weights], test_cudf[cols], test_cudf[target])
    
    val_data = cudf_data.sample(10)
    val_target = val_data[target]
    val_data = val_data[cols]
    pred = model.predict(val_data)
    print(pred.get())
    print(val_target.values)

    model = TorchBasedLogisticRegression(data_size = 3, output_size = 4, **params)
    model.fit(train_data[cols].values, train_data[target].values, train_data[weights].values, test_data[cols].values, test_data[target].values)
    
    val_data = val_data.to_pandas()
    val_target = val_target.to_pandas()

    pred = model.predict(val_data.values)
    print(pred)
    print(val_target.values)

    print("HELLO")
