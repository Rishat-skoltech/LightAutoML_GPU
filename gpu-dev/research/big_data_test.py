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
    #cudf.set_allocator("managed")
    ds = dask_cudf.read_parquet("./data/big_dummy.parquet", npartitions=3)
    ds = ds.append(dask_cudf.read_parquet("./data/big_dummy.parquet", npartitions=3))
    ds = ds.append(dask_cudf.read_parquet("./data/big_dummy.parquet", npartitions=3))
    print(len(ds), len(ds.columns))
    print("Number of columns with nans before:", ds.map_partitions(count_nans).compute().sum())
    #print(ds.compute())
    print("starting the timer!")
    t1 = timeit.default_timer()
    res = ds.map_partitions(rr.FillnaMedian_rapids)
    res = res.map_partitions(count_nans)
    print("Number of columns with nans after:", res.compute().sum())
    #print(res.compute())
    t2 = timeit.default_timer()
    print("Wall time RAPIDS:" , t2-t1, "seconds.")

def generate_dummy_parquet():
    TARGET_NAME, cols, data = uu.generate_data(6000000, 60, 0, 0, 40000)
    ddf = dd.from_pandas(data, npartitions=70)
    ddf.to_parquet('./data/big_dummy.parquet', write_index=False)

if __name__ == '__main__':
    if (not os.path.exists('./data/big_dummy.parquet')):
        if (len(sys.argv)>1 and sys.argv[1] != "generate_data"):
            print("this script generates 2.9 GB of dummy data, then it's going to be read 3 times to take up 5.7 GB of memory, if you okay with it launch it again with arg 'generate_data'")
            exit()
        else:
            generate_dummy_parquet()
    run()
