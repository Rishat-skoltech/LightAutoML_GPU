from lightautoml.automl.presets.gpu.tabular_gpu_presets import TabularUtilizedAutoML_gpu
from lightautoml.tasks import Task

import pandas as pd
import numpy as np

import cudf

from os import listdir

files = listdir('.')
csv_files = [elem for elem in files if elem.endswith(".csv")]

print(csv_files)
csv_files.append('ashrae-energy-prediction/processed_train.csv')
csv_files.append('ieee-fraud-detection/processed_train.csv')
csv_files.append('bnp-paribas-cardif-claims-management/train.csv')
csv_files.append('porto-seguro-safe-driver-prediction/train.csv')
csv_files.append('springleaf-marketing-response/train.csv')
csv_files.append('talkingdata-adtracking-fraud-detection/train.csv')

data_info = {
    
    # OPENML
    
    'covertype': {
        'path': 'openml/covertype.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'class_map': {x: x - 1 for x in range(1, 8)},
        'read_csv_params': {'na_values': '?'}
    },
    
    'albert': {
        'path': 'openml/albert.csv',
        'target': 'class',
        'task_type': 'binary',
        'read_csv_params': {'na_values': '?'}
    },
    
    'higgs': {
        'path': 'openml/higgs.csv',
        'target': 'class',
        'task_type': 'binary',
        'read_csv_params': {'na_values': '?'}
    },
    
    'guillermo': {
        'path': 'openml/guillermo.csv',
        'target': 'class',
        'task_type': 'binary',
        'read_csv_params': {'na_values': '?'}
    },
    
    'bank-marketing': {
        'path': 'openml/bank-marketing.csv',
        'target': 'Class',
        'task_type': 'binary',
        'class_map': {x: x - 1 for x in range(1, 3)},
        'read_csv_params': {'na_values': '?'}
    },
    
    'numerai28.6': {
        'path': 'openml/numerai28.6.csv',
        'target': 'attribute_21',
        'task_type': 'binary',
        'read_csv_params': {'na_values': '?'}
    },
    
    'volkert': {
        'path': 'openml/volkert.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'read_csv_params': {'na_values': '?'}
    },
    
    'adult': {
        'path': 'openml/adult.csv',
        'target': 'class',
        'task_type': 'binary',
        'class_map': {' <=50K': 0, ' >50K': 1},
        'read_csv_params': {'na_values': '?'}
    },
    
    'MiniBooNE': {
        'path': 'openml/MiniBooNE.csv',
        'target': 'signal',
        'task_type': 'binary',
        'class_map': {False: 0, True: 1},
        'read_csv_params': {'na_values': '?'}
    },
    
    'dilbert': {
        'path': 'openml/dilbert.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'read_csv_params': {'na_values': '?'}
    },
    
    'riccardo': {
        'path': 'openml/riccardo.csv',
        'target': 'class',
        'task_type': 'binary',
        'read_csv_params': {'na_values': '?'}
    },
    
    'shuttle': {
        'path': 'openml/shuttle.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'class_map': {x: x - 1 for x in range(1, 8)},
        'read_csv_params': {'na_values': '?'}
    },
    
    'KDDCup09_appetency': {
        'path': 'openml/KDDCup09_appetency.csv',
        'target': 'APPETENCY',
        'task_type': 'binary',
        'class_map': {-1: 0, 1: 1},
        #'read_csv_params': {'na_values': '?'}
    },
    
    'Fashion-MNIST': {
        'path': 'openml/Fashion-MNIST.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'read_csv_params': {'na_values': '?'}
    },
    
    'connect-4': {
        'path': 'openml/connect-4.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'read_csv_params': {'na_values': '?'}
    },
    
    'airlines': {
        'path': 'openml/airlines.csv',
        'target': 'Delay',
        'task_type': 'binary',
        'read_csv_params': {'na_values': '?'}
    },
    
    'jannis': {
        'path': 'openml/jannis.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'read_csv_params': {'na_values': '?'}
    },
    
    'nomao': {
        'path': 'openml/nomao.csv',
        'target': 'Class',
        'task_type': 'binary',
        'class_map': {x: x - 1 for x in range(1, 3)},
        'read_csv_params': {'na_values': '?'}
    },
    
    'Amazon_employee_access': {
        'path': 'openml/Amazon_employee_access.csv',
        'target': 'target',
        'task_type': 'binary',
        'read_csv_params': {'na_values': '?'}
    },
    
    'robert': {
        'path': 'openml/robert.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'read_csv_params': {'na_values': '?'}
    },
    
    'aps_failure': {
        'path': 'openml/aps_failure.csv',
        'target': 'class',
        'task_type': 'binary',
        'class_map': {'neg': 0, 'pos': 1},
        'read_csv_params': {'na_values': '?'}
    },
    
    'jungle_chess_2pcs_raw_endgame_complete': {
        'path': 'openml/jungle_chess_2pcs_raw_endgame_complete.csv',
        'target': 'class',
        'task_type': 'multiclass',
        'class_map': {'w': 0, 'b': 1, 'd': 2},
        'read_csv_params': {'na_values': '?'}
    },
    
    # KAGGLE ....
    'ashrae-energy-prediction/processed_train': {
        'path': 'ashrae-energy-prediction/processed_train.csv',
        'target': 'meter_reading',
        'task_type': 'reg'
    },
    
    'ieee-fraud-detection/processed_train': {
        'path': 'ieee-fraud-detection/processed_train.csv',
        'target': 'isFraud',
        'task_type': 'binary',
    },
    
    'bnp-paribas-cardif-claims-management/train': {
        'path': 'bnp-paribas-cardif-claims-management/train.csv',
        'target': 'target',
        'task_type': 'binary',
        'drop': ['ID']
    },
    
    'porto-seguro-safe-driver-prediction/train': {
        'path': 'porto-seguro-safe-driver-prediction/train.csv',
        'target': 'target',
        'task_type': 'binary',
        'drop': ['id']
    },
    
    'springleaf-marketing-response/train': {
        'path': 'springleaf-marketing-response/train.csv',
        'target': 'target',
        'task_type': 'binary',
        'drop': ['ID']
    },  
    
    'talkingdata-adtracking-fraud-detection/train': {
        'path': 'talkingdata-adtracking-fraud-detection/train.csv',
        'target': 'is_attributed',
        'task_type': 'binary',
        'drop': ['attributed_time']
    }, 
}



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
                'jungle_chess_2pcs_raw_endgame_complete.csv' : 'class',
                'ashrae-energy-prediction/processed_train.csv' : 'meter_reading',
                'ieee-fraud-detection/processed_train.csv' : 'isFraud',
                'bnp-paribas-cardif-claims-management/train.csv' : 'target',
                'porto-seguro-safe-driver-prediction/train.csv' : 'target',
                'springleaf-marketing-response/train.csv' : 'target',
                'talkingdata-adtracking-fraud-detection/train.csv' : 'is_attributed',
                'jobs_train.csv' : 'target'
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
             'jungle_chess_2pcs_raw_endgame_complete.csv' : 'multiclass',
             'ashrae-energy-prediction/processed_train.csv' : 'reg',
             'ieee-fraud-detection/processed_train.csv' : 'binary',
             'bnp-paribas-cardif-claims-management/train.csv' : 'binary',
             'porto-seguro-safe-driver-prediction/train.csv' : 'binary',
             'springleaf-marketing-response/train.csv' : 'binary',
             'talkingdata-adtracking-fraud-detection/train.csv' : 'binary',
             'jobs_train.csv' : 'binary'
             }



if __name__ == "__main__":

    #cluster = LocalCUDACluster(rmm_managed_memory=True, CUDA_VISIBLE_DEVICES="0,1",
    #                           protocol="ucx", enable_nvlink=True,
    #                           memory_limit="8GB")

    #client = Client(cluster)
    #client.run(cudf.set_allocator, "managed")

    cudf.set_allocator("managed")

    for i, cur_file in enumerate(csv_files):
        #if cur_file in ['dilbert.csv', 'volkert.csv', 'KDDCup09_appetency.csv', 'talkingdata-adtracking-fraud-detection/train.csv', 'springleaf-marketing-response/train.csv', 'bnp-paribas-cardif-claims-management/train.csv']:
        if cur_file in ['KDDCup09_appetency.csv']:
            pass
        else:
            continue
        print("####################################")
        print(cur_file)
        print(task_types[cur_file])
        print(TARGETS_DICT[cur_file])
        print("####################################")


        data = pd.read_csv(cur_file)
        cur_data_info = data_info[cur_file[:-4]]

        if 'read_csv_params' in cur_data_info:
            for col in data.columns:
                if data[col].isin(['?']).any():
                    data[col] = data[col].replace('?', np.nan).astype(np.float32)

        if 'drop' in cur_data_info:
            data.drop(cur_data_info['drop'], axis=1, inplace=True)

        if 'class_map' in cur_data_info:
            data[cur_data_info['target']] = data[cur_data_info['target']].map(cur_data_info['class_map']).values
            assert data[cur_data_info['target']].notnull().all(), 'Class mapping is set unproperly'

        # run automl
        # this is for small amounts of data
        automl = TabularUtilizedAutoML_gpu(task=Task(task_types[cur_file], device="mgpu"), timeout=3600)#, client=client)
        #automl = TabularAutoML_gpu(task=Task(task_types[cur_file], device="gpu"), timeout=3600)#, client=client)
        # this is for bigger amounts of data
        #automl = TabularAutoML_gpu(task=Task(task_types[cur_file], device="mgpu"))

        oof_predictions = automl.fit_predict(data, roles={'target':TARGETS_DICT[cur_file]}, verbose=2)

        print("NOW DOING PREDICTIONS")

        te_pred = automl.predict(data)

        print(type(oof_predictions))
        print(type(te_pred))
        print("####################################")
        print("####################################")
        print()
