#!/usr/bin/env python
# coding: utf-8
import logging

from time import perf_counter

import numpy as np
import pandas as pd
import dask_cudf
import cudf

from lightautoml.tasks import Task
from lightautoml.reader.cudf_reader import CudfReader
from lightautoml.reader.daskcudf_reader import DaskCudfReader
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.dataset.roles import DropRole, DatetimeRole, CategoryRole, TargetRole

from numba import jit
import string

################################
# Features:
# - working with cudf.DataFrame
# - working with dask_cudf.DataFrame
# - using CudfReader
# - using DaskCudfReader
################################

RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))

@jit(nopython=True)
def gen_cols(n_cols):
    cols = [""]*n_cols
    for i in range(n_cols):
        cols[i] = "col_" + str(i)
    return cols

@jit(nopython=True)
def gen_string_data(n, n_str):
    string_data = [""]*n_str*n
    for i in range(n):
        for j in range(n_str):
            nchars = np.random.randint(1, 10)
            string_data[i*n_str+j] = "".join(np.random.choice(RANDS_CHARS, nchars))
    return string_data

def generate_data(n, n_num, n_cat, n_date, n_str, nan_amount, num_categories):
    print("Generating dummy data")
    n_cols = n_num+n_cat+n_str+n_date
    cols = gen_cols(n_cols)

    data = np.random.random((n, n_num))*100-50
    data.ravel()[np.random.choice(data.size, nan_amount, replace=False)] = np.nan

    category_data = np.random.randint(0, num_categories, (n, n_cat))
    string_data = gen_string_data(n, n_str)

    string_data = np.reshape(string_data, (n, n_str))

    date_data = np.random.randint(0, 1000,
                               (n, n_date)).astype(np.dtype("timedelta64[D]")) \
                               + np.datetime64("2018-01-01")

    data = np.append(data, string_data, axis=1)
    data = np.append(data, date_data.astype("U"), axis=1)
    data = np.append(data, category_data, axis=1)
    data = pd.DataFrame(data, columns=cols)
    data[cols[-1]] = data[cols[0-1]].astype(np.int)
    print("Size of the dummy data:",
          round(data.memory_usage(deep=True).sum()/1024./1024.,4), "MB.")
    return cols[-1], cols[:-1], data

def test_cudf_daskcudf_readers_basic():
    #logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s',
    #                    level=logging.DEBUG)
    task = Task("binary")

    print("#### TEST-01 ####")
    target, _, data = generate_data(n=40000, n_num=200, n_cat=20, n_date=0,
                                    n_str=0, nan_amount=0, num_categories=2)
    cudf_data = cudf.DataFrame.from_pandas(data)
    daskcudf_data = dask_cudf.from_cudf(cudf_data, npartitions=4)
    pd_reader = PandasToPandasReader(task, cv=5, random_state=42,
                                     advanced_roles=False)
    cudf_reader = CudfReader(task, cv=5, random_state=42)
    daskcudf_reader = DaskCudfReader(task, cv=5, random_state=42)

    start = perf_counter()
    pd_dataset = pd_reader.fit_read(data, target=data[target])
    print("PandasToPandasReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", pd_dataset.data.shape)
    start = perf_counter()
    cudf_dataset = cudf_reader.fit_read(cudf_data, target=cudf_data[target])
    print("CudfReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", cudf_dataset.data.shape)
    start = perf_counter()
    daskcudf_dataset = daskcudf_reader.fit_read(daskcudf_data,
                                                target=daskcudf_data[target])
    print("DaskCudfReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is",
          daskcudf_dataset.data.compute().shape)

    print("Are outputs from padnas and cudf close?",
          np.allclose(pd_dataset.data.to_numpy().astype(np.float),
                      cudf_dataset.data.to_pandas().to_numpy().astype(np.float)))
    print("Are outputs from pandas and dask_cudf close?",
          np.allclose(pd_dataset.data.to_numpy().astype(np.float),
              daskcudf_dataset.data.compute().to_pandas().to_numpy().astype(np.float)))
    print("TEST-01 FINISHED")

    print("#### TEST-02 ####")
    target, _, data = generate_data(n=10, n_num=2, n_cat=2, n_date=2,
                                    n_str=2, nan_amount=6, num_categories=2)
    cudf_data = cudf.DataFrame.from_pandas(data)
    daskcudf_data = dask_cudf.from_cudf(cudf_data, npartitions=4)
    pd_reader = PandasToPandasReader(task, cv=5, random_state=42, advanced_roles=False)
    cudf_reader = CudfReader(task, cv=5, random_state=42)
    daskcudf_reader = DaskCudfReader(task, cv=5, random_state=42)

    start = perf_counter()
    pd_dataset = pd_reader.fit_read(data, target=data[target])
    print("PandasToPandasReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", pd_dataset.data.shape)
    start = perf_counter()
    cudf_dataset = cudf_reader.fit_read(cudf_data, target=cudf_data[target])
    print("CudfReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", cudf_dataset.data.shape)
    start = perf_counter()
    daskcudf_dataset = daskcudf_reader.fit_read(daskcudf_data,
                                                target=daskcudf_data[target])
    print("DaskCudfReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", daskcudf_dataset.data.compute().shape)
    print("TEST-02 FINISHED")

    print("#### TEST-03 ####")
    print("reading sampled_app_train.csv...", end="")
    data = pd.read_csv("../example_data/test_data_files/sampled_app_train.csv")
    data["BIRTH_DATE"] = np.datetime64("2018-01-01") \
                       + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    data["EMP_DATE"] = np.datetime64("2018-01-01")\
                     + np.clip(data["DAYS_EMPLOYED"], None, 0)\
                       .astype(np.dtype("timedelta64[D]"))
    print("done")
    print("fit_read using predefined roles:")
    roles = {DropRole(): ["FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_18"],
             TargetRole(): "TARGET", CategoryRole(): ["SK_ID_CURR", "AMT_DatetimeRole"],
             DatetimeRole(): ["BIRTH_DATE", "EMP_DATE"]}
    print(roles)
    cudf_data = cudf.DataFrame.from_pandas(data)
    daskcudf_data = dask_cudf.from_cudf(cudf_data, npartitions=4)
    pd_reader = PandasToPandasReader(task, cv=5, random_state=42,
                                     advanced_roles=False)
    cudf_reader = CudfReader(task, cv=5, random_state=42)
    daskcudf_reader = DaskCudfReader(task, cv=5, random_state=42)

    start = perf_counter()
    pd_dataset = pd_reader.fit_read(data, target=data["TARGET"], roles=roles)
    print("PandasToPandasReader fit_read time:", perf_counter()-start)
    start = perf_counter()
    cudf_dataset = cudf_reader.fit_read(cudf_data,
                   target=cudf_data["TARGET"], roles=roles)
    print("CudfReader fit_read time:", perf_counter()-start)
    start = perf_counter()
    daskcudf_dataset = daskcudf_reader.fit_read(daskcudf_data,
                       target=daskcudf_data["TARGET"], roles=roles)
    print("DaskCudfReader fit_read time:", perf_counter()-start)

    print("Shape of PandasDataset", pd_dataset.data.shape)
    print("Shape of CudfDataset", cudf_dataset.data.shape)
    print("Shape of DaskCudfDataset", daskcudf_dataset.data.compute().shape)

    print("PandasReader dropped features:", pd_reader.dropped_features)
    print("CudfReader dropped features:", cudf_reader.dropped_features)
    print("DaskCudfReader dropped features:", daskcudf_reader.dropped_features)
    print("TEST-03 FINISHED")

    print("#### TEST-04 ####")
    print("reading sampled_app_train.csv...", end="")
    data = pd.read_csv("../example_data/test_data_files/sampled_app_train.csv")
    data["BIRTH_DATE"] = np.datetime64("2018-01-01") \
                       + data["DAYS_BIRTH"].astype(np.dtype("timedelta64[D]"))
    data["EMP_DATE"] = np.datetime64("2018-01-01") \
                     + np.clip(data["DAYS_EMPLOYED"], None, 0)\
                       .astype(np.dtype("timedelta64[D]"))
    print("done")
    print("fit_read without any predefined roles:")
    cudf_data = cudf.DataFrame.from_pandas(data)
    daskcudf_data = dask_cudf.from_cudf(cudf_data, npartitions=4)
    pd_reader = PandasToPandasReader(task, cv=5, random_state=42,
                                     advanced_roles=False)
    cudf_reader = CudfReader(task, cv=5, random_state=42)
    daskcudf_reader = DaskCudfReader(task, cv=5, random_state=42)

    start = perf_counter()
    pd_dataset = pd_reader.fit_read(data, target=data["TARGET"])
    print("PandasToPandasReader fit_read time:", perf_counter()-start)
    start = perf_counter()
    cudf_dataset = cudf_reader.fit_read(cudf_data, target=cudf_data["TARGET"])
    print("CudfReader fit_read time:", perf_counter()-start)
    start = perf_counter()
    daskcudf_dataset = daskcudf_reader.fit_read(daskcudf_data,
                                       target=daskcudf_data["TARGET"])
    print("DaskCudfReader fit_read time:", perf_counter()-start)

    print("Shape of PandasDataset", pd_dataset.data.shape)
    print("Shape of CudfDataset", cudf_dataset.data.shape)
    print("Shape of DaskCudfDataset", daskcudf_dataset.data.compute().shape)

    print("PandasReader dropped features:", pd_reader.dropped_features)
    print("CudfReader dropped features:", cudf_reader.dropped_features)
    print("DaskCudfReader dropped features:", daskcudf_reader.dropped_features)

    #for col in daskcudf_dataset.data.columns:
    #    print(cudf_dataset.roles[col].name, pd_dataset.roles[col].name,
    #          daskcudf_dataset.roles[col].name)
    print("TEST-04 FINISHED")

if __name__ == "__main__":
    test_cudf_daskcudf_readers_basic()
