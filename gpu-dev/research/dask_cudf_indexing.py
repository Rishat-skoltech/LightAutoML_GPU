import cupy as cp
import cudf
import dask_cudf
import dask.dataframe as dd

import utils.utils as uu
import rapids_impl.rapids_impl as rr

import numpy as np
import pandas as pd

import timeit
import os
import sys

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

def count_nans(cudf_data):
    count = 0
    for feature in cudf_data.columns:
        if cudf_data[feature].has_nulls:
            count += 1
    return count
    
def run():
    cluster = LocalCUDACluster()
    client = Client(cluster)
    client.run(cudf.set_allocator, "managed")
    TARGET_NAME, cols, data = uu.generate_data(10000, 25, 0, 0, 0)
    cu_data = cudf.from_pandas(data)
    ds = dask_cudf.from_cudf(cu_data, npartitions=3)
    print(ds.compute())
    print(ds.loc[[9997, 1, 3, 9995]].compute())
    print(ds.divisions)

if __name__ == '__main__':
    run()
