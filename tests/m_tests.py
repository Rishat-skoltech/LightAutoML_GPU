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

from lightautoml.reader.hybrid_reader import HybridReader
from lightautoml.reader.cudf_reader import CudfReader
from lightautoml.reader.base import PandasToPandasReader


from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.ml_algo.boost_cb import BoostCB
from lightautoml.ml_algo.boost_xgb_gpu import BoostXGB, BoostXGB_dask

from lightautoml.ml_algo.linear_gpu import LinearL1CD_gpu, LinearL1CD_mgpu#, LinearLBFGS_gpu
from lightautoml.ml_algo.linear_sklearn import LinearL1CD#, LinearLBFGS

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
        
    data['target_mu'] = pd.Series(np.random.randint(0, 4, n)).astype('i')
    data['target_bi'] = pd.Series(np.random.randint(0, 2, n)).astype('i')
    data['target_re'] = pd.Series(np.random.random((n))*10).astype('f')

    print("Shape of the dummy data:", data.shape)
    print("Size of the dummy data:",
          round(data.memory_usage(deep=True).sum()/1024./1024.,4), "MB.")
    return 'target', cols, data


def test():
    cluster = LocalCUDACluster(rmm_managed_memory=True, CUDA_VISIBLE_DEVICES="0",
                               protocol="ucx", enable_nvlink=True,
                               memory_limit="8GB")
    print("dashboard:", cluster.dashboard_link)
    client = Client(cluster)
    client.run(cudf.set_allocator, "managed")

    task_re_mgpu = Task("reg", device="mgpu")
    task_re_gpu = Task("reg", device="gpu")
    task_re = Task("reg", device="cpu")

    task_bi_mgpu = Task("binary", metric="accuracy", device="mgpu")
    task_bi_gpu = Task("binary", device="gpu")
    task_bi = Task("binary", device="cpu")

    task_mu_mgpu = Task("multiclass", device="mgpu")
    task_mu_gpu = Task("multiclass", device="gpu")
    task_mu = Task("multiclass", device="cpu")

    _, _, data = generate_data(n=40, n_num=3, n_cat=0, n_date=0,
                                    n_str=0, max_n_cat=10)

    pd_reader_re = PandasToPandasReader(task_re, cv=5, random_state=42, n_jobs=1, advanced_roles=True)
    pd_reader_bi = PandasToPandasReader(task_bi, cv=5, random_state=42, n_jobs=1, advanced_roles=True)
    pd_reader_mu = PandasToPandasReader(task_mu, cv=5, random_state=42, n_jobs=1, advanced_roles=True)

    cudf_reader_re = CudfReader(task_re_gpu, device_num=0, cv=5, random_state=42, n_jobs=1, advanced_roles=True)
    cudf_reader_bi = CudfReader(task_bi_gpu, device_num=0, cv=5, random_state=42, n_jobs=1, advanced_roles=True)
    cudf_reader_mu = CudfReader(task_mu_gpu, device_num=0, cv=5, random_state=42, n_jobs=1, advanced_roles=True)

    hybrid_reader_re = HybridReader(task_re_mgpu, num_cpu_readers=1, num_gpu_readers=0, gpu_ratio=0.5, output='mgpu', cv=5, random_state=42)
    hybrid_reader_bi = HybridReader(task_bi_mgpu, num_cpu_readers=1, num_gpu_readers=0, gpu_ratio=0.5, output='mgpu', cv=5, random_state=42)
    hybrid_reader_mu = HybridReader(task_mu_mgpu, num_cpu_readers=1, num_gpu_readers=0, gpu_ratio=0.5, output='mgpu', cv=5, random_state=42)

    output_bi_mgpu = hybrid_reader_bi.fit_read(data, target=data['target_bi'])
    output_bi_mgpu.data = output_bi_mgpu.data.astype('f')

    output_re_mgpu = hybrid_reader_re.fit_read(data, target=data['target_re'])
    output_re_mgpu.data = output_re_mgpu.data.astype('f')

    output_mu_mgpu = hybrid_reader_mu.fit_read(data, target=data['target_mu'])
    output_mu_mgpu.data = output_mu_mgpu.data.astype('f')

    output_re_gpu = cudf_reader_re.fit_read(data, target=data['target_re'])
    output_re_gpu.data = output_re_gpu.data.astype('f')

    output_bi_gpu = cudf_reader_bi.fit_read(data, target=data['target_bi'])
    output_bi_gpu.data = output_bi_gpu.data.astype('f')
 
    output_mu_gpu = cudf_reader_mu.fit_read(data, target=data['target_mu'])
    output_mu_gpu.data = output_mu_gpu.data.astype('f')
 
    output_re = pd_reader_re.fit_read(data, target=data['target_re'])
    output_re.data = output_re.data.astype('f')

    output_bi = pd_reader_bi.fit_read(data, target=data['target_bi'])
    output_bi.data = output_bi.data.astype('f')
 
    output_mu = pd_reader_mu.fit_read(data, target=data['target_mu'])
    output_mu.data = output_mu.data.astype('f')
 
 
    folds_it_re_gpu = FoldsIterator_gpu(output_re_gpu)
    folds_it_bi_gpu = FoldsIterator_gpu(output_bi_gpu)
    folds_it_mu_gpu = FoldsIterator_gpu(output_mu_gpu)

    folds_it_re_mgpu = FoldsIterator_gpu(output_re_mgpu)
    folds_it_bi_mgpu = FoldsIterator_gpu(output_bi_mgpu)
    folds_it_mu_mgpu = FoldsIterator_gpu(output_mu_mgpu)

    folds_it_re = FoldsIterator(output_re)
    folds_it_bi = FoldsIterator(output_bi)
    folds_it_mu = FoldsIterator(output_mu)

    #################################################################################################
    '''cpu_solver = LinearLBFGS()
    res = cpu_solver.fit_predict(folds_it_re)
    cpu_solver = LinearLBFGS()
    res = cpu_solver.fit_predict(folds_it_bi)
    cpu_solver = LinearLBFGS()
    res = cpu_solver.fit_predict(folds_it_mu)
    print('cpu fin')
    gpu_solver = LinearLBFGS_gpu()
    res_gpu = gpu_solver.fit_predict(folds_it_bi_gpu)
    gpu_solver = LinearLBFGS_gpu()
    res_gpu = gpu_solver.fit_predict(folds_it_re_gpu)
    gpu_solver = LinearLBFGS_gpu()
    res_gpu = gpu_solver.fit_predict(folds_it_mu_gpu)
    print('gpu fin')'''
    ##################################################################################################
    cpu_solver = LinearL1CD()
    res = cpu_solver.fit_predict(folds_it_re)
    cpu_solver = LinearL1CD()
    res = cpu_solver.fit_predict(folds_it_bi)
    cpu_solver = LinearL1CD()
    res = cpu_solver.fit_predict(folds_it_mu)
    print('cpu fin')
    gpu_solver = LinearL1CD_gpu()
    res_gpu = gpu_solver.fit_predict(folds_it_bi_gpu)
    gpu_solver = LinearL1CD_gpu()
    res_gpu = gpu_solver.fit_predict(folds_it_re_gpu)
    gpu_solver = LinearL1CD_gpu()
    res_gpu = gpu_solver.fit_predict(folds_it_mu_gpu)
    print('gpu fin')
    mgpu_solver = LinearL1CD_mgpu(client)
    res_mgpu = mgpu_solver.fit_predict(folds_it_re_mgpu)
    '''print("2")
    mgpu_solver = LinearL1CD_mgpu(client)
    res_mgpu = mgpu_solver.fit_predict(folds_it_bi_mgpu)
    print("1")
    mgpu_solver = LinearL1CD_mgpu(client)
    res_mgpu = mgpu_solver.fit_predict(folds_it_mu_mgpu)
    print("0")
    print('mgpu fin')'''
    #################################################################################################
    cpu_solver = BoostLGBM()
    res = cpu_solver.fit_predict(folds_it_bi)
    cpu_solver = BoostLGBM()
    res = cpu_solver.fit_predict(folds_it_mu)
    cpu_solver = BoostLGBM()
    res = cpu_solver.fit_predict(folds_it_re)
    print('cpu fin LGBM')
    cpu_solver = BoostCB()
    res = cpu_solver.fit_predict(folds_it_bi)
    cpu_solver = BoostCB()
    res = cpu_solver.fit_predict(folds_it_mu)
    cpu_solver = BoostCB()
    res = cpu_solver.fit_predict(folds_it_re)
    print('cpu fin CB')
    gpu_solver = BoostXGB()
    res_gpu = gpu_solver.fit_predict(folds_it_mu_gpu)
    gpu_solver = BoostXGB()
    res_gpu = gpu_solver.fit_predict(folds_it_bi_gpu)
    gpu_solver = BoostXGB()
    res_gpu = gpu_solver.fit_predict(folds_it_re_gpu)
    print('gpu fin XGB')
    mgpu_solver = BoostXGB_dask(client)
    res_mgpu = mgpu_solver.fit_predict(folds_it_bi_mgpu)
    mgpu_solver = BoostXGB_dask(client)
    res_mgpu = mgpu_solver.fit_predict(folds_it_mu_mgpu)
    mgpu_solver = BoostXGB_dask(client)
    res_mgpu = mgpu_solver.fit_predict(folds_it_re_mgpu)
    print('mgpu fin XGB')
    ################################################################################################
if __name__ == "__main__":
    test()

