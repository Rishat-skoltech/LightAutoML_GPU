import os
from time import time
from dask_ml.model_selection import train_test_split
from sklearn.model_selection import train_test_split as train_test_cpu
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
import cupy as cp
from sklearn.linear_model import Lasso
from sklearn.metrics import average_precision_score, accuracy_score
import sys

import numpy as np
from cuml.linear_model import Lasso as cudfLasso
from time import perf_counter
from numba import jit
import numpy as np
import pandas as pd
import string

RANDS_CHARS = np.array(list(string.ascii_letters + string.digits), dtype=(np.str_, 1))

@jit(nopython=True)
def gen_cols(N_cols):
    cols = ['']*N_cols
    for i in range(N_cols):
        cols[i] = "col_" + str(i)
    return cols

@jit(nopython=True)
def gen_string_data(N, N_str):
    string_data = ['']*N_str*N
    for i in range(N):
        for j in range(N_str):
            nchars = np.random.randint(1, 10)
            string_data[i*N_str+j] = ''.join(np.random.choice(RANDS_CHARS, nchars))
    return string_data

def generate_data(N, N_num, N_cat, N_str, nan_amount):
    N_cols = N_num+N_cat+N_str
    cols = gen_cols(N_cols)

    numeric_data = np.random.random((N, N_num))*100-50
    numeric_data.ravel()[np.random.choice(numeric_data.size, nan_amount, replace=False)] = np.nan

    category_data = np.random.randint(0, 2, (N, N_cat))
    string_data = gen_string_data(N, N_str)

    string_data = np.reshape(string_data, (N,N_str))

    numeric_data = np.append(numeric_data, category_data, axis=1)
    numeric_data = np.append(numeric_data, string_data, axis=1)

    data = pd.DataFrame(numeric_data, columns=cols)
    print("Size of the dummy data:", round(data.memory_usage(deep=True).sum()/1024./1024.,4), "MB.")
    return cols[-1], cols[:-1], data
    
def main(client):
    TARGET_NAME, cols, data = generate_data(150000, 100, 1, 0, 0)
    
    y = data[TARGET_NAME]
    X = data[data.columns.difference([TARGET_NAME])]
    X_train,X_test,y_train,y_test = train_test_cpu(X,y, test_size=0.4,random_state=42)
    Lasso_model = Lasso()    
    start = perf_counter()
    Lasso_model.fit(X_train, y_train)
    print("CPU LASSO TIME TRAIN:", perf_counter()-start)
    start = perf_counter()
    res = Lasso_model.predict(X_test)
    print("CPU LASSO TIME INFER:", perf_counter()-start)
    print("SCORE:", average_precision_score(y_test, res))
    
    cdf = cudf.from_pandas(data)
    dask_cdf = dask_cudf.from_cudf(cdf, npartitions=4)
    print(dask_cdf.compute().shape)
    y = dask_cdf[dask_cdf.columns[-1]]
    X = dask_cdf[dask_cdf.columns.difference([dask_cdf.columns[-1]])]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_train, X_valid, y_train, y_valid = client.persist(
        [X_train, X_valid, y_train, y_valid]
    )
    wait([X_train, X_valid, y_train, y_valid])

    t1 = perf_counter()
    regression = daskLasso(client=client)
    res = regression.fit(X_train, y_train)
    t2 = perf_counter()
    y_res = res.predict(X_valid).compute()
    t3 = perf_counter()
    y_val_rest = y_valid.reset_index(drop=True)#.compute()
    acc = accuracy.accuracy_score(y_val_rest.compute(), y_res)
    print("Regression accuracy score:", acc)
    print("Lasso training time:", t2-t1)
    print("Lasso inference time:", t3-t2)

    ls = cudfLasso(alpha=0.1)
    X_train = X_train.compute()
    X_valid = X_valid.compute()
    y_train = y_train.compute()
    y_valid = y_valid.compute()

    start=perf_counter()
    result_lasso = ls.fit(X_train, y_train)
    print("Lasso training cudf:", perf_counter()-start)
    start = perf_counter()
    preds = result_lasso.predict(X_valid)
    print("Lasso inference cudf:", perf_counter()-start)
    acc = accuracy.accuracy_score(preds, y_valid)
    print("Regression accuracy score:", acc)

if __name__ == "__main__":
   #with LocalCUDACluster(CUDA_VISIBLE_DEVICES="0,1,2,3", protocol="ucx", enable_nvlink=True, dashboard_address=None, rmm_managed_memory=True, memory_limit="4GB") as cluster:
   with LocalCUDACluster(CUDA_VISIBLE_DEVICES="0", protocol="ucx", enable_nvlink=True, dashboard_address=None, rmm_managed_memory=True, memory_limit="8GB") as cluster:
       print("dashboard:", cluster.dashboard_link)
       with Client(cluster) as client:
           client.run(cudf.set_allocator, "managed")
           main(client)
