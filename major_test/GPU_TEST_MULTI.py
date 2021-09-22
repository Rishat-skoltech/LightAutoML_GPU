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
from lightautoml.reader.daskcudf_reader import DaskCudfReader

# Standard python libraries
import logging
import os
import time
import requests
logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.INFO)

# Installed libraries
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import torch

# Imports from our package
from lightautoml.automl.base import AutoML



from lightautoml.pipelines.features.lgb_pipeline_gpu import LGBSimpleFeatures_gpu, LGBAdvancedPipeline_gpu
from lightautoml.pipelines.features.linear_pipeline_gpu import LinearFeatures_gpu
from lightautoml.ml_algo.boost_xgb_gpu import BoostXGB, BoostXGB_dask

from lightautoml.ml_algo.linear_gpu import LinearLBFGS_gpu
from lightautoml.ml_algo.tuning.optuna import OptunaTuner

from lightautoml.pipelines.ml.base import MLPipeline
from lightautoml.pipelines.selection.importance_based import ModelBasedImportanceEstimator, ImportanceCutoffSelector

from lightautoml.automl.blend_gpu import WeightedBlender_gpu

from lightautoml.utils.profiler import Profiler
from lightautoml.utils.timer import PipelineTimer

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

def test_pipeline(client):

    target, _, data = generate_data(n=40, n_num=3, n_cat=2, n_date=5,
                                    n_str=5, max_n_cat=10)
                                    
    train_data, test_data = train_test_split(data, 
                                         test_size=TEST_SIZE, 
                                         stratify=data[TARGET_NAME], 
                                         random_state=RANDOM_STATE)
    
    test_daskcudf = cudf.DataFrame.from_pandas(test_data, nan_as_null=False)
    test_daskcudf = dask_cudf.from_cudf(test_daskcudf, npartitions=1)
    
    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(N_THREADS)
    
    timer = PipelineTimer(600, mode=2)
    timer_gbm = timer.get_task_timer('gbm') # Get task timer from pipeline timer 

    feat_sel_0 = LGBSimpleFeatures_gpu()
    mod_sel_0 = BoostXGB_dask(client, timer=timer_gbm)
    imp_sel_0 = ModelBasedImportanceEstimator()
    selector_0 = ImportanceCutoffSelector(feat_sel_0, mod_sel_0, imp_sel_0, cutoff=0, )
    feats_gbm_0 = LGBAdvancedPipeline_gpu(top_intersections=4, 
                                  output_categories=True, 
                                  feats_imp=imp_sel_0)

    timer_gbm_0 = timer.get_task_timer('gbm')
    timer_gbm_1 = timer.get_task_timer('gbm')
    
    gbm_0 = BoostXGB_dask(client, timer=timer_gbm_0)
    gbm_1 = BoostXGB(timer=timer_gbm_1)

    tuner_0 = OptunaTuner(n_trials=20, timeout=30, fit_on_holdout=True)
    gbm_lvl0 = MLPipeline([
            (gbm_0, tuner_0),
            gbm_1
        ],
        pre_selection=selector_0,
        features_pipeline=feats_gbm_0, 
        post_selection=None
    )
    
    feats_reg_0 = LinearFeatures_gpu(output_categories=True, 
                             sparse_ohe='auto')

    timer_reg = timer.get_task_timer('reg')
    reg_0 = LinearLBFGS_gpu(timer=timer_reg)

    reg_lvl0 = MLPipeline([
            reg_0
        ],
        pre_selection=None,
        features_pipeline=feats_reg_0, 
        post_selection=None
    )
    task = Task('multiclass', metric = 'accuracy', device='mgpu')
    
    reader = HybridReader(task = task, num_cpu_readers=1, num_gpu_readers=1,
                              gpu_ratio=0.5, output='mgpu', npartitions= 1,
                              samples = None, max_nan_rate = 1,
                              max_constant_rate = 1, advanced_roles = True,
                              drop_score_co = -1, n_jobs = 1, compute=True)
                              
    blender = WeightedBlender_gpu()
    automl = AutoML(reader=reader, levels=[
        [gbm_lvl0, reg_lvl0]
    ], timer=timer, blender=blender, skip_conn=False)
    
    oof_pred = automl.fit_predict(train_data, roles={'target': TARGET_NAME})

    logging.info('oof_pred:\n{}\nShape = {}'.format(oof_pred.data.compute(), oof_pred.shape))
    
    #test_pred = automl.predict(test_daskcudf)
    #logging.debug('Prediction for test data:\n{}\nShape = {}'
    #          .format(test_pred, test_pred.shape))

    #logging.info('Check scores...')
    logging.info('OOF score: {}'.format(log_loss(train_data[TARGET_NAME].values, oof_pred.data.compute().values.get())))
    #logging.info('TEST score: {}'.format(log_loss(test_daskcudf[TARGET_NAME].compute().values.get(), test_pred.data.compute().get())))

if __name__ == "__main__":
    with LocalCUDACluster(rmm_managed_memory=True, CUDA_VISIBLE_DEVICES="0",
                               protocol="ucx", enable_nvlink=True,
                               memory_limit="8GB") as cluster:
        print("dashboard:", cluster.dashboard_link)
        with Client(cluster) as client:
            client.run(cudf.set_allocator, "managed")

            test_pipeline(client)

    #client = 1
    #test_pipeline(client)
