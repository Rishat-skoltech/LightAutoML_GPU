import cudf
import numpy as np
import dask_cudf
import cuml
from memory_profiler import profile
from cuml.preprocessing import StandardScaler
from xgboost import dask as dxgb
import os

#@profile
def FillnaMedian_rapids(cudf_data):
    data_gpu = cudf.DataFrame()
    for feature in cudf_data.columns:
        if cudf_data[feature].has_nulls:
            data_gpu[feature] = cudf_data[feature].fillna(cudf_data[feature].median().astype('float32'))
        else:
            data_gpu[feature] = cudf_data[feature]
    return data_gpu

#@profile
def StandardScaler_rapids(cudf_data):
    scaler = StandardScaler()
    data_gpu = cudf.DataFrame()
    with cuml.using_output_type('cudf'):
        scaler.fit(cudf_data)
        data_gpu = scaler.transform(cudf_data)
    return data_gpu
    
def to_parquet(path, filename) -> str:
   print("starting to save the parquet file...")
   dirpath = path
   parquetfile = filename+".parquet"
   parquet_path = os.path.join(dirpath, parquetfile)
   if os.path.exists(parquet_path):
       return parquet_path
   csvfile = filename+".csv"
   csv_path = os.path.join(dirpath, csvfile)
   colnames = ["label"] + ["feature-%02d" % i for i in range(1, 29)]
   df = dask_cudf.read_csv(csv_path, header=None, names=colnames, dtype=np.float32)
   df.to_parquet(parquet_path)
   print("finished")
   return parquet_path
   
def XGBoost_dask(client, X_train, y_train, X_valid, y_valid):
    Xy = dxgb.DaskDeviceQuantileDMatrix(client, X_train, y_train)
    Xy_valid = dxgb.DaskDMatrix(client, X_valid, y_valid)
    return dxgb.train(
        client,
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "gpu_hist",
        },
        Xy,
        evals=[(Xy_valid, "Valid")],
        verbose_eval=False,
        num_boost_round=100,
    )["booster"]
        
def SelectColumns_rapids(cudf_data, cols):
    return cudf_data[cols]
    
    
    
    
    
    
    
    
    
    
