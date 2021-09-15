#!/usr/bin/env python
# coding: utf-8
from time import perf_counter

import numpy as np
import pandas as pd
import dask_cudf
import cudf
import cupy as cp
import random

from lightautoml.tasks import Task
from lightautoml.pipelines.features.lgb_pipeline_gpu import LGBSimpleFeatures_gpu, LGBAdvancedPipeline_gpu
from lightautoml.pipelines.features.linear_pipeline_gpu import LinearFeatures_gpu

from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures, LGBAdvancedPipeline
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures

from lightautoml.reader.hybrid_reader import HybridReader

from lightautoml.validation.np_iterators import FoldsIterator
from lightautoml.validation.gpu_iterators import FoldsIterator_gpu

from numba import jit
import string

RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))

N_THREADS = 8 # threads cnt for lgbm and linear models
N_FOLDS = 5 # folds cnt for AutoML
RANDOM_STATE = 42 # fixed random state for various reasons
TEST_SIZE = 0.2 # Test size for metric check
TIMEOUT = 600 # Time in seconds for automl run
TARGET_NAME = 'target' # Target column name


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


def test():
    task = Task("multiclass")
    target, _, data = generate_data(n=40, n_num=3, n_cat=2, n_date=5,
                                    n_str=5, max_n_cat=10)

    hybrid_reader = HybridReader(task, num_cpu_readers=1, num_gpu_readers=0, gpu_ratio=0.5, output='mgpu', cv=5, random_state=42)
    output = hybrid_reader.fit_read(data, target=data[target])
    print(type(output))
    gpu_output = output.to_cudf()
    cpu_output = gpu_output.to_pandas()

    #################################################################################################
    LGBA_gpu = LGBAdvancedPipeline_gpu()
    LGBA = LGBAdvancedPipeline()
    
    mgpu_final = LGBA_gpu.create_pipeline(output).fit_transform(output).data.compute().values.get()
    gpu_final = LGBA_gpu.create_pipeline(gpu_output).fit_transform(gpu_output).data.values.get()
    cpu_final = LGBA.create_pipeline(cpu_output).fit_transform(cpu_output).data

    print(np.allclose(gpu_final, cpu_final))
    print(np.allclose(mgpu_final, gpu_final))
    print(gpu_final.shape)
    print(mgpu_final.shape)
    print(cpu_final.shape)
    ################################################################################################
    LGBS_gpu = LGBSimpleFeatures_gpu()
    LGBS = LGBSimpleFeatures()

    mgpu_final = LGBS_gpu.create_pipeline(output).fit_transform(output).data.compute().values.get()
    gpu_final = LGBS_gpu.create_pipeline(gpu_output).fit_transform(gpu_output).data.values.get()
    cpu_final = LGBS.create_pipeline(cpu_output).fit_transform(cpu_output).data

    print(np.allclose(gpu_final, cpu_final))
    print(np.allclose(mgpu_final, gpu_final))
    print(gpu_final.shape)
    print(mgpu_final.shape)
    print(cpu_final.shape)
    ################################################################################################
    LF_gpu = LinearFeatures_gpu()
    LF = LinearFeatures()

    mgpu_final = LF_gpu.create_pipeline(output).fit_transform(output).data.compute().values.get()
    gpu_final = LF_gpu.create_pipeline(gpu_output).fit_transform(gpu_output).data.values.get()
    cpu_final = LF.create_pipeline(cpu_output).fit_transform(cpu_output).data

    print(np.allclose(gpu_final, cpu_final))
    print(np.allclose(mgpu_final, gpu_final))
    print(gpu_final.shape)
    print(mgpu_final.shape)
    print(cpu_final.shape)
    ################################################################################################
    ################################################################################################

    folds_it_gpu = FoldsIterator_gpu(gpu_output)
    folds_it_mgpu = FoldsIterator_gpu(output)
    folds_it = FoldsIterator(cpu_output)

    for n, (idx, train, valid) in enumerate(folds_it_gpu):
        print(n)
        print(idx)
        print(train.shape)
        print(valid.shape)
        print("###############")
    print("###############################################")
    for n, (idx, train, valid) in enumerate(folds_it_mgpu):
        print(n)
        print(idx)
        print(train.data.compute().shape)
        print(valid.data.compute().shape)
        print("###############")
    print("###############################################")
    for n, (idx, train, valid) in enumerate(folds_it):
        print(n)
        print(idx)
        print(train.shape)
        print(valid.shape)
        print("###############")
    print("###############################################")

if __name__ == "__main__":
    test()

