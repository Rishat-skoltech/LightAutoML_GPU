import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from scipy import sparse

from time import perf_counter
from lightautoml.reader.cudf_reader import CudfReader
from lightautoml.reader.daskcudf_reader import DaskCudfReader
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.reader.hybrid_reader import HybridReader

from lightautoml.dataset.roles import ColumnRole

from lightautoml.tasks import Task

import cudf
import dask_cudf

from numba import cuda
import cupy as cp

from lightautoml.transformers.base import SequentialTransformer, UnionTransformer


from lightautoml.transformers import numeric_gpu, categorical_gpu, datetime_gpu
from lightautoml.transformers import numeric, categorical, datetime

from lightautoml.pipelines.utils import get_columns_by_role

from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import os

def test(client):
    data = pd.read_csv('./application_train.csv')

    data['BIRTH_DATE'] = (np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))).astype(str)
    data['EMP_DATE'] = (np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))
                       ).astype(str)
    #data['BIRTH_DATE'] = np.datetime64('2018-01-01') + data['DAYS_BIRTH'].astype(np.dtype('timedelta64[D]'))
    #data['EMP_DATE'] = np.datetime64('2018-01-01') + np.clip(data['DAYS_EMPLOYED'], None, 0).astype(np.dtype('timedelta64[D]'))

    data['constant'] = 1
    data['allnan'] = np.nan

    data.drop(['DAYS_BIRTH',  'DAYS_EMPLOYED'], axis = 1, inplace = True)

    data = data.sample(frac=0.001)

    gpu_data = cudf.DataFrame.from_pandas(data, nan_as_null=False)
    dd_data = dask_cudf.from_cudf(gpu_data, npartitions=1)

    task = task = Task('binary',)
    adv_roles = True
    reader = PandasToPandasReader(task, advanced_roles=adv_roles, n_jobs=1)
    gpu_reader = CudfReader(task, advanced_roles=adv_roles, n_jobs=1)
    #dd_reader = DaskCudfReader(task, advanced_roles=adv_roles, n_jobs=1, compute=False, npartitions=1)
    
    hy_reader = HybridReader(task, num_cpu_readers=1, num_gpu_readers=1, gpu_ratio=0.5, output='mgpu', advanced_roles=adv_roles, npartitions=1, n_jobs=1)

    print(type(dd_data))
    st = perf_counter()
    ds = reader.fit_read(data, roles = {'target': 'TARGET'})
    print(perf_counter() - st)
    st = perf_counter()
    gpu_ds = gpu_reader.fit_read(gpu_data, roles = {'target': 'TARGET'})
    print(perf_counter() - st)
    st = perf_counter()
    dd_ds = hy_reader.fit_read(data, roles = {'target': 'TARGET'})
    print(perf_counter() - st)
    print("after readers")
    #print(ds.roles)

    trf = categorical.LabelEncoder()
    gpu_trf = categorical_gpu.LabelEncoder_gpu()
    dd_trf = categorical_gpu.LabelEncoder_gpu()

    print("after setting label encoder")
    cats = ds[:, get_columns_by_role(ds, 'Category')]
    gpu_cats = gpu_ds[:, get_columns_by_role(gpu_ds, 'Category')]
    dd_cats = dd_ds[:, get_columns_by_role(dd_ds, 'Category')]

    print(cats.shape, gpu_cats.shape, dd_cats.shape)
    print("after choosing columns")
    #print(get_columns_by_role(ds, 'Category'))

    st = perf_counter()
    enc = trf.fit_transform(cats)
    print(perf_counter() - st, "cpu labelencoder")
    st = perf_counter()
    enc = gpu_trf.fit_transform(gpu_cats)
    print(perf_counter() - st, "gpu labelencoder")
    #cuda.synchronize()
    st = perf_counter()
    enc = dd_trf.fit_transform(dd_cats)
    print(perf_counter() - st, "mgpu labelencoder")
    #cuda.synchronize()
    print("after fit transforming")
    trf = SequentialTransformer(
        [categorical.LabelEncoder(), categorical.TargetEncoder()]
    )

    gpu_trf = SequentialTransformer(
        [categorical_gpu.LabelEncoder_gpu(), categorical_gpu.TargetEncoder_gpu()]
    )

    dd_trf = SequentialTransformer(
        [categorical_gpu.LabelEncoder_gpu(), categorical_gpu.TargetEncoder_gpu()]
    )

    print("after craeting sequential transformers")
    st = perf_counter()
    enc = trf.fit_transform(cats)
    print(perf_counter() - st, "cpu sequential")
    print("2")
    st = perf_counter()
    enc = gpu_trf.fit_transform(gpu_cats)
    #cuda.synchronize()
    print(perf_counter() - st, "gpu sequentia")
    print("1")
    st = perf_counter()
    enc = dd_trf.fit_transform(dd_cats)
    #cuda.synchronize()
    print(perf_counter() - st, "mgpu sequential")
    print("0")
    print(data.shape)
    print("FINISHED")


if __name__ == "__main__":
    cluster = LocalCUDACluster(rmm_managed_memory=True, CUDA_VISIBLE_DEVICES="0",
                               protocol="ucx", enable_nvlink=True,
                               memory_limit="8GB")

    client = Client(cluster)
    # client.run(cudf.set_allocator, "managed")
    # client.run(os.getpid)
    test(client)

