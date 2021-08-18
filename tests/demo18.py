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
        data[cols[nn+i]] = pd.Series(category_data[:,i]).astype('i')
        
    
    
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
    print("THE DATA:")
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
    
    ################################################################################
    filler_mgp = datetime_gpu.DateSeasons_gpu()
    filler_gpu = datetime_gpu.DateSeasons_gpu()
    filler_cpu = datetime.DateSeasons()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,dat_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,dat_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,dat_roles])
    
    print("DateSeasons:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))
    #################################################################################

    filler_mgp = datetime_gpu.TimeToNum_gpu()
    filler_gpu = datetime_gpu.TimeToNum_gpu()
    filler_cpu = datetime.TimeToNum()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,dat_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,dat_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,dat_roles])
    
    print("TimeToNum:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))
    ##################################################################################

    filler_mgp = datetime_gpu.BaseDiff_gpu(base_names=['col_9', 'col_10'], diff_names=['col_11'])
    filler_gpu = datetime_gpu.BaseDiff_gpu(base_names=['col_9', 'col_10'], diff_names=['col_11'])
    filler_cpu = datetime.BaseDiff(base_names=['col_9', 'col_10'], diff_names=['col_11'])
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,dat_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,dat_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,dat_roles])
    
    print("BaseDiff:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))
    ##################################################################################

    filler_mgp = numeric_gpu.NaNFlags_gpu()
    filler_gpu = numeric_gpu.NaNFlags_gpu()
    filler_cpu = numeric.NaNFlags()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,num_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,num_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,num_roles])
    
    print("NaNFlags:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))
    ##################################################################################

    filler_mgp = numeric_gpu.FillnaMedian_gpu()
    filler_gpu = numeric_gpu.FillnaMedian_gpu()
    filler_cpu = numeric.FillnaMedian()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,num_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,num_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,num_roles])
    
    print("FillnaMedian:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))
    ##################################################################################

    filler_mgp = numeric_gpu.LogOdds_gpu()
    filler_gpu = numeric_gpu.LogOdds_gpu()
    filler_cpu = numeric.LogOdds()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,num_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,num_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,num_roles])
    
    print("LogOdds:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.values.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))
    ##################################################################################

    filler_mgp = numeric_gpu.StandardScaler_gpu()
    filler_gpu = numeric_gpu.StandardScaler_gpu()
    filler_cpu = numeric.StandardScaler()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,num_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,num_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,num_roles])
    
    print("StandardScaler:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))
   ##################################################################################

    filler_mgp = numeric_gpu.QuantileBinning_gpu()
    filler_gpu = numeric_gpu.QuantileBinning_gpu()
    filler_cpu = numeric.QuantileBinning()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,num_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,num_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,num_roles])
    
    print("QuantileBinning:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))
    ##################################################################################

    filler_mgp = categorical_gpu.LabelEncoder_gpu()
    filler_gpu = categorical_gpu.LabelEncoder_gpu()
    filler_cpu = categorical.LabelEncoder()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,str_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,str_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,str_roles])
    
    print("LabelEncoder:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))

    ##################################################################################

    filler_mgp = categorical_gpu.OHEEncoder_gpu(make_sparse=False)
    filler_gpu = categorical_gpu.OHEEncoder_gpu(make_sparse=False)
    filler_cpu = categorical.OHEEncoder(make_sparse=False)
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,cat_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,cat_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,cat_roles])
    
    print("OHEEncoder:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))

    ##################################################################################

    filler_mgp = categorical_gpu.FreqEncoder_gpu()
    filler_gpu = categorical_gpu.FreqEncoder_gpu()
    filler_cpu = categorical.FreqEncoder()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,str_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,str_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,str_roles])
    
    print("FreqEncoder:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))

    ##################################################################################

    filler_mgp = categorical_gpu.OrdinalEncoder_gpu()
    filler_gpu = categorical_gpu.OrdinalEncoder_gpu()
    filler_cpu = categorical.OrdinalEncoder()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,cat_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,cat_roles])
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,cat_roles])
    
    print("OrdinalEncoder:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))

    ##################################################################################
    
    filler_mgp = categorical_gpu.TargetEncoder_gpu()
    filler_gpu = categorical_gpu.TargetEncoder_gpu()
    filler_cpu = categorical.TargetEncoder()
    
    filled_cpu = filler_cpu.fit_transform(pd_dataset[:,cat_roles])
    filled_gpu = filler_gpu.fit_transform(cudf_dataset[:,cat_roles])
    
    daskcudf_dataset.data = daskcudf_dataset.data.reset_index(drop=True)
    filled_mgp = filler_mgp.fit_transform(daskcudf_dataset[:,cat_roles])
    
    print("TargetEncoder:")
    print("Are outputs close: pandas vs cudf", np.allclose(filled_gpu.data.get(), filled_cpu.data))
    print("Are outputs close: pandas vs dask_cudf", np.allclose(filled_mgp.data.compute().values.get(), filled_cpu.data))

if __name__ == "__main__":
    test_transformers()
    
