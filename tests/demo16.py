#!/usr/bin/env python
# coding: utf-8
import os
from time import perf_counter

import numpy as np
import pandas as pd
import dask_cudf
import dask.dataframe as dd
import cudf

from lightautoml.tasks import Task
from lightautoml.reader.cudf_reader import CudfReader

from numba import jit
import string
import sys

################################
# Features:
# - working with cudf.DataFrame
# - writing dummy data to file
# - reading data from file many times to fill memory
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

def generate_dummy_parquet():
    _, _, data = generate_data(n=50000, n_num=300, n_cat=10, n_date=0,
                               n_str=0, nan_amount=0, num_categories=2)
    ddf = dd.from_pandas(data, npartitions=30)
    ddf.to_parquet("./data/big_dummy.parquet", write_index=True)

def test_large_data_cudf():
    #logging.basicConfig(format='[%(asctime)s] (%(levelname)s): %(message)s',
    #                    level=logging.DEBUG)
    if not os.path.exists("./data/big_dummy.parquet"):
        if (len(sys.argv)==1 or (len(sys.argv)>1 and sys.argv[1] != "generate_data")):
            print("""this script generates 1097 MB of dummy data files,
                  then it's going to be read 5 times to take about 5 GB, 
                  if you okay with it launch it again with arg 'generate_data'""")
            sys.exit()
        else:
            generate_dummy_parquet()

    cudf.set_allocator("managed")
    task = Task('binary')
    print("#### TEST-05 ####")
    print("reading dummy data from file...", end="")
    daskcudf_data = dask_cudf.read_parquet("./data/big_dummy.parquet")
    daskcudf_data = daskcudf_data.append(dask_cudf.read_parquet("./data/big_dummy.parquet"))
    daskcudf_data = daskcudf_data.append(dask_cudf.read_parquet("./data/big_dummy.parquet"))
    daskcudf_data = daskcudf_data.append(dask_cudf.read_parquet("./data/big_dummy.parquet"))
    daskcudf_data = daskcudf_data.append(dask_cudf.read_parquet("./data/big_dummy.parquet"))
    print("done")

    cudf_data = daskcudf_data.compute()
    cudf_reader = CudfReader(task, cv=5, random_state=42, advanced_roles=False)
    start = perf_counter()
    cudf_dataset = cudf_reader.fit_read(cudf_data,
                                        target=cudf_data[cudf_data.columns[-1]])
    print("CudfReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", cudf_dataset.data.shape)

    print("CudfDataset shape", cudf_dataset.data.shape)
    print("CudfReader dropped features:", cudf_reader.dropped_features)
    print("TEST-05 FINISEHD")

if __name__ == "__main__":
    test_large_data_cudf()
