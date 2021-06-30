import os
from time import time

from dask import dataframe as dd
from dask_cuda import LocalCUDACluster
from distributed import Client, wait
import dask_cudf
import cudf
from dask_ml.model_selection import train_test_split
from dask_ml.metrics import log_loss
from cuml.metrics import log_loss as log_loss_cudf
from cuml.metrics import accuracy
from cuml.dask.linear_model import Lasso as daskLasso

from xgboost import dask as dxgb
from time import perf_counter
import rapids_impl.rapids_impl as rr

def main(client):
    t0 = perf_counter()
    if (not os.path.exists('./data/HIGGS.parquet')):
        if (not os.path.exists('./data/HIGGS.csv')):
            print("you need to download HIGGS data set, call it HIGGS.csv (it's gonna be 8GB) and place it in the 'data' folder")
            exit()
        else:
            rr.to_parquet('./data', 'HIGGS')
    df = dask_cudf.read_parquet("./data/HIGGS.parquet")
    t1 = perf_counter()
    df = df.map_partitions(rr.FillnaMedian_rapids)
    wait([df])
    t2 = perf_counter()
    y = df["label"]
    X = df[df.columns.difference(["label"])]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_train, X_valid, y_train, y_valid = client.persist(
        [X_train, X_valid, y_train, y_valid]
    )
    wait([X_train, X_valid, y_train, y_valid])
    t3 = perf_counter()
    booster = rr.XGBoost_dask(client, X_train, y_train, X_valid, y_valid)
    t4 = perf_counter()
    predt = dxgb.predict(client, booster, X_valid)#.compute()
    acc = log_loss_cudf(y_valid.compute(), predt.compute())
    print("XGB logloss score:", acc)
    t5 = perf_counter()
    features = dict(sorted(booster.get_fscore().items(), key=lambda item: item[1], reverse=True))
    important_f = list(features.keys())[:10]    
    X_train=  X_train.map_partitions(rr.SelectColumns_rapids, important_f)#.compute()
    X_valid = X_valid.map_partitions(rr.SelectColumns_rapids, important_f)#.compute()
    X_train = X_train.map_partitions(rr.StandardScaler_rapids)
    X_valid = X_valid.map_partitions(rr.StandardScaler_rapids)
    wait([X_train, X_valid])
    t6 = perf_counter()
    regression = daskLasso(client=client)
    res = regression.fit(X_train, y_train)
    t7 = perf_counter()
    y_res = res.predict(X_valid)#.compute()
    y_val_rest = y_valid.reset_index(drop=True)#.compute()
    acc = accuracy.accuracy_score(y_val_rest.compute(), y_res.compute())
    print("Regression accuracy score:", acc)
    t8 = perf_counter()

    print("read parquet",t1-t0)
    print("fillna:", t2-t1)
    print("prepare data", t3-t2)
    print("XGBoost training time:", t4-t3)
    print("XGBoost inference time:", t5-t4)
    print("Feature select + scale time:", t6-t5)
    print("Lasso training time:", t7-t6)
    print("Lasso inference time:", t8-t7)
    return

if __name__ == "__main__":
   #with LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1,2,3", protocol="ucx", enable_nvlink=True, dashboard_address=None, rmm_managed_memory=True, memory_limit="4GB") as cluster:
   with LocalCUDACluster(CUDA_VISIBLE_DEVICES="0", protocol="ucx", enable_nvlink=True, dashboard_address=None, rmm_managed_memory=True, memory_limit="4GB") as cluster:
       print("dashboard:", cluster.dashboard_link)
       with Client(cluster) as client:
           client.run(cudf.set_allocator, "managed")
           main(client)
           print("finished")
