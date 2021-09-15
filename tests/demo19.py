#!/usr/bin/env python
# coding: utf-8
from time import perf_counter

import numpy as np
import pandas as pd
import dask_cudf
import cudf
import cupy as cp
import random

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from lightautoml.tasks import Task
from lightautoml.reader.cudf_reader import CudfReader
from lightautoml.reader.daskcudf_reader import DaskCudfReader
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.dataset.roles import DropRole, DatetimeRole, CategoryRole, TargetRole

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
    for row, col in random.sample(ix, int(round(.1*len(ix)))):
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

def test_multiclass():

    #cluster = LocalCUDACluster(rmm_managed_memory=True, CUDA_VISIBLE_DEVICES="0",
    #                           protocol="ucx", enable_nvlink=True,
    #                           memory_limit="8GB")
    #print("dashboard:", cluster.dashboard_link)
    #client = Client(cluster)
    #client.run(cudf.set_allocator, "managed")

    task = Task("multiclass")

    target, _, data = generate_data(n=40, n_num=3, n_cat=2, n_date=5,
                                    n_str=5, max_n_cat=10)
                                    
    print(data)
    cudf_data = cudf.DataFrame.from_pandas(data, nan_as_null=False)
    daskcudf_data = dask_cudf.from_cudf(cudf_data, npartitions=1)
    
    pd_reader = PandasToPandasReader(task, cv=5, random_state=42,
                                     advanced_roles=True)
    cudf_reader = CudfReader(task, cv=5, random_state=42, advanced_roles=True)
    daskcudf_reader = DaskCudfReader(task, cv=5, random_state=42,
                                     advanced_roles=True, compute=True)
    
    start = perf_counter()
    pd_dataset = pd_reader.fit_read(data, target=data[target])
    print("PandasToPandasReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", pd_dataset.data.shape)
    start = perf_counter()
    daskcudf_dataset = daskcudf_reader.fit_read(daskcudf_data,
                                                target=daskcudf_data[target])
    print("DaskCudfReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is",
          daskcudf_dataset.data.compute().shape)    
    start = perf_counter()
    cudf_dataset = cudf_reader.fit_read(cudf_data, target=cudf_data[target])
    print("CudfReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", cudf_dataset.data.shape)
    

if __name__ == "__main__":
    test_multiclass()
