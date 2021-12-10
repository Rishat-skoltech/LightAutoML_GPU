from lightautoml.automl.presets.tabular_gpu_presets import TabularAutoML_gpu
from lightautoml.tasks import Task
from lightautoml.dataset.roles import TargetRole

import pandas as pd
import numpy as np

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import cudf

from os import listdir

files = listdir('.')
csv_files = [elem for elem in files if elem.endswith(".csv")]

TARGETS_DICT = {'covertype.csv' : 'class' , 'albert.csv' : 'class',
                'higgs.csv' : 'class', 'guillermo.csv' : 'class',
                'bank-marketing.csv' : 'Class',
                'numerai28.6.csv' : 'attribute_21', 'volkert.csv' : 'class',
                'adult.csv' : 'class', 'MiniBooNE.csv' : 'signal',
                'dilbert.csv' : 'class', 'riccardo.csv' : 'class',
                'shuttle.csv' : 'class', 'KDDCup09_appetency.csv' : 'APPETENCY',
                'Fashion-MNIST.csv' : 'class', 'connect-4.csv' : 'class',
                'airlines.csv' : 'Delay', 'jannis.csv' : 'class',
                'nomao.csv' : 'Class', 'Amazon_employee_access.csv' : 'target',
                'robert.csv' : 'class', 'aps_failure.csv' : 'class',
                'jungle_chess_2pcs_raw_endgame_complete.csv' : 'class'
               }
task_types = {'covertype.csv' : 'multiclass' , 'albert.csv' : 'binary',
             'higgs.csv' : 'binary', 'guillermo.csv' : 'binary',
             'bank-marketing.csv' : 'binary',
              'numerai28.6.csv' : 'binary', 'volkert.csv' : 'multiclass',
              'adult.csv' : 'binary', 'MiniBooNE.csv' : 'binary',
              'dilbert.csv' : 'multiclass', 'riccardo.csv' : 'binary',
              'shuttle.csv' : 'multiclass', 'KDDCup09_appetency.csv' : 'binary',
              'Fashion-MNIST.csv' : 'multiclass', 'connect-4.csv' : 'multiclass',
              'airlines.csv' : 'binary', 'jannis.csv' : 'multiclass',
              'nomao.csv' : 'binary', 'Amazon_employee_access.csv' : 'binary',
              'robert.csv' : 'multiclass', 'aps_failure.csv' : 'binary',
              'jungle_chess_2pcs_raw_endgame_complete.csv' : 'multiclass'
             }

if __name__ == "__main__":

    cluster = LocalCUDACluster(rmm_managed_memory=True, CUDA_VISIBLE_DEVICES="0,1",
                               protocol="ucx", enable_nvlink=True,
                               memory_limit="8GB")

    client = Client(cluster)
    client.run(cudf.set_allocator, "managed")

    print(csv_files)
    #cur_file = csv_files[1]
    cur_file = "jobs_train.csv"

    data = pd.read_csv(cur_file)
    for col in data.columns:
        if data[col].isin(['?']).any():
            data[col] = data[col].replace('?', np.nan).astype(np.float32)
    # run automl
    # this is for small amounts of data
    automl = TabularAutoML_gpu(task=Task('binary', device="mgpu"))#, client = client)
    #automl = TabularAutoML_gpu(task=Task(task_types[cur_file], device="gpu"))
    # this is for bigger amounts of data
    #automl = TabularAutoML_gpu(task=Task(task_types[cur_file], device="mgpu"))

    oof_predictions = automl.fit_predict(data, roles={TargetRole():'target'}, verbose=2)

    #print("NOW DOING PREDICTIONS")

    #te_pred = automl.predict(data)

    #print(type(oof_predictions))
    #print(type(te_pred))
