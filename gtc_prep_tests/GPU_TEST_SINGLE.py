#!/usr/bin/env python
# coding: utf-8
from time import perf_counter

import numpy as np
import pandas as pd
import cudf
import cupy as cp
import random

from lightautoml.tasks import Task
from lightautoml.reader.cudf_reader import CudfReader

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
from lightautoml.ml_algo.boost_xgb_gpu import BoostXGB

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

def test_pipeline():
    #target, _, data = generate_data(n=200, n_num=3, n_cat=2, n_date=5,
    #                                n_str=5, max_n_cat=10)
                                    
    data = pd.read_csv('./jobs_train.csv')

    #data['BIRTH_DATE'] = (np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))).astype(str)
    #data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
    #                   ).astype(str)
    #data['BIRTH_DATE'] = np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))
    #data['EMP_DATE'] = np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))

    #data['constant'] = 1
    #data['allnan'] = np.nan

    #data.drop(['DAYS_BIRTH',  'DAYS_EMPLOYED'], axis = 1, inplace = True)
    print(len(data))
    #data[TARGET_NAME] = pd.Series(np.random.randint(0, 4, len(data)))

    #data = data.sample(100000).reset_index(drop=True)
    #data = pd.concat([data, data]).reset_index(drop=True)
    #data = pd.concat([data, data])
    #data = pd.concat([data, data])
    #print(data.shape)

    data_cudf = cudf.from_pandas(data, nan_as_null=False)
    
    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(N_THREADS)

    timer = PipelineTimer(600, mode=2)
    timer_gbm = timer.get_task_timer('gbm') # Get task timer from pipeline timer

    feat_sel_0 = LGBSimpleFeatures_gpu()    
    mod_sel_0 = BoostXGB(timer=timer_gbm)
    imp_sel_0 = ModelBasedImportanceEstimator()
    selector_0 = ImportanceCutoffSelector(feat_sel_0, mod_sel_0, imp_sel_0, cutoff=0, )
    
    feats_gbm_0 = LGBAdvancedPipeline_gpu(top_intersections=4, 
                                  output_categories=True, 
                                  feats_imp=imp_sel_0)
    timer_gbm_0 = timer.get_task_timer('gbm')
    timer_gbm_1 = timer.get_task_timer('gbm')

    gbm_0 = BoostXGB(timer=timer_gbm_0)
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

    task = Task('binary', metric = 'logloss', device='gpu') 
    reader = CudfReader(task = task, device_num = 0, samples = 100000,  max_nan_rate = 1,
                              max_constant_rate = 1, advanced_roles = True,
                              drop_score_co = -1, n_jobs = 1)
                              
    blender = WeightedBlender_gpu()
    automl = AutoML(reader=reader, levels=[
        [gbm_lvl0, reg_lvl0]
    ], timer=timer, skip_conn=False, blender=blender)
    
    st = perf_counter()
    oof_pred = automl.fit_predict(data_cudf, roles={'target': TARGET_NAME})
    print(perf_counter() - st, "TOTAL TIME SINGLE GPU")

    logging.info('oof_pred:\n{}\nShape = {}'.format(oof_pred, oof_pred.shape))

    '''cudf_test  = data_cudf.sample(10000).reset_index(drop=True)
    st = perf_counter()
    test_res = automl.predict(cudf_test)
    print(perf_counter() - st, "predict time")'''

if __name__ == "__main__":
    cudf.set_allocator("managed")
    test_pipeline()
