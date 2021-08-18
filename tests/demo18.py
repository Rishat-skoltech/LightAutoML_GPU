#!/usr/bin/env python
# coding: utf-8
from time import perf_counter

import numpy as np
import pandas as pd
import dask_cudf
import cudf
import cupy as cp
import random
from copy import copy
from time import perf_counter

from lightautoml.tasks import Task

from lightautoml.dataset.np_pd_dataset import PandasDataset
from lightautoml.dataset.cp_cudf_dataset import CudfDataset
from lightautoml.dataset.daskcudf_dataset import DaskCudfDataset

from lightautoml.dataset.roles import *
from lightautoml.dataset.utils import roles_parser

from numba import jit
import string


from lightautoml.transformers import numeric_gpu, categorical_gpu, datetime_gpu
from lightautoml.transformers import numeric, categorical, datetime

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
                 "backup", "bandwidth", "binary", "bit", "byte",
                 "bitmap", "blog", "bookmark", "boot", "broadband",
                 "browser" , "buffer", "bug"]
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
    for row, col in random.sample(ix, int(round(.2*len(ix)))):
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
        
    data['target'] = pd.Series(np.random.randint(0, 2, n)).astype('i')

    print("Shape of the dummy data:", data.shape)
    print("Size of the dummy data:",
          round(data.memory_usage(deep=True).sum()/1024./1024.,4), "MB.")
    return 'target', cols, data

def test_transformers():
    

    task = Task("binary")
    target, _, data = generate_data(n=10, n_num=3, n_cat=3, n_date=3,
                                    n_str=3, max_n_cat=10)
    
    data['__fold__'] = np.random.randint(0, 5, len(data))
    
    print(data)
    

    num_roles = ['col_0', 'col_1', 'col_2']    
    cat_roles = ['col_3', 'col_4', 'col_5']
    str_roles = ['col_6', 'col_7', 'col_8']
    dat_roles = ['col_9', 'col_10', 'col_11']
    check_roles = {
        TargetRole(): 'target',
        CategoryRole(dtype=int, label_encoded=True): cat_roles,
        CategoryRole(dtype=str): str_roles,
        NumericRole(np.float32): num_roles,
        DatetimeRole(seasonality=['y', 'm', 'wd']): dat_roles,
        FoldsRole(): '__fold__'
    }


    cudf_data = cudf.DataFrame.from_pandas(data, nan_as_null=False)
    daskcudf_data = dask_cudf.from_cudf(cudf_data, npartitions=2)
    
    pd_dataset = PandasDataset(data, roles_parser(check_roles), task=task)
    cudf_dataset = CudfDataset(cudf_data, roles_parser(check_roles), task=task)
    daskcudf_dataset = DaskCudfDataset(daskcudf_data, roles_parser(check_roles), task=task)
    
    filler_mgp = datetime_gpu.DateSeasons_gpu()   #base_names=['col_4', 'col_5'], diff_names=['col_6'])
    filler_gpu = datetime_gpu.DateSeasons_gpu()   #base_names=['col_4', 'col_5'], diff_names=['col_6'])
    filler_cpu = datetime.DateSeasons()           #base_names=['col_4', 'col_5'], diff_names=['col_6'])
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,dat_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,dat_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,dat_roles])
    
    print(filled_mgp.data.compute())
    print(filled_gpu.data)
    print(filled_cpu)
    #for i in range(len(filled_cpu.data)):
    #    print(filled_cpu.data[i], filled_gpu.data[i])
    
if __name__ == "__main__":
    test_transformers()
    
