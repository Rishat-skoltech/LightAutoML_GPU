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
from lightautoml.reader.hybrid_reader import HybridReader

# Standard python libraries
import logging
import os
import time
import requests
logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s', level=logging.INFO)

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




def create_pipeline(task, num_threads=4, random_state=42):
    
    np.random.seed(random_state)
    torch.set_num_threads(num_threads)

    feat_sel_0 = LGBSimpleFeatures_gpu()    
    mod_sel_0 = BoostXGB()
    imp_sel_0 = ModelBasedImportanceEstimator()
    selector_0 = ImportanceCutoffSelector(feat_sel_0, mod_sel_0, imp_sel_0, cutoff=0, )

    # features
    feats_gbm_0 = LGBAdvancedPipeline_gpu(top_intersections=4, 
                                      output_categories=True, 
                                      feats_imp=imp_sel_0)

    # xgb
    gbm_0 = BoostXGB()
    feats_reg_0 = LinearFeatures_gpu(output_categories=True, 
                                 sparse_ohe=False)

    gbm_lvl0 = MLPipeline([
                gbm_0
            ],
            pre_selection=selector_0,
            features_pipeline=feats_gbm_0, 
            post_selection=None
        )

    # linear
    feats_reg_0 = LinearFeatures_gpu(output_categories=True, 
                                 sparse_ohe=False)

    reg_0 = LinearLBFGS_gpu()

    reg_lvl0 = MLPipeline([
            reg_0
        ],
        pre_selection=None,
        features_pipeline=feats_reg_0, 
        post_selection=None
    )


    reader = HybridReader(task = task,  samples = 100000,  max_nan_rate = 1,
                          max_constant_rate = 1, advanced_roles = True, 
                          num_cpu_readers=num_threads, num_gpu_readers=1, gpu_ratio=0.7, output='gpu',
                          drop_score_co = -1, n_jobs = 1)

    blender = WeightedBlender_gpu()
    automl = AutoML(reader=reader, levels=[
        [gbm_lvl0, reg_lvl0]
    ],  skip_conn=False, blender=blender)
    
    return automl

if __name__ == "__main__":
    data = pd.read_csv('./jobs_train.csv')
    task = Task('binary',  device='gpu') 
    automl = create_pipeline(task)
    roles={'target': 'target'}
    oof_pred = automl.fit_predict(data, roles=roles)




