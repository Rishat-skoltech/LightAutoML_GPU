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

from numba import cuda


def str_concat(x, y, out):
    for i, (a, b) in enumerate(zip(x, y)):
        out[i] = a + b

if __name__ == '__main__':
    TARGET_NAME, cols, data = uu.generate_data(10000, 0, 0, 3, 0)
    cdf = cudf.from_pandas(data)
    print(cdf.sample(10))
    
    res = cdf.apply_rows(str_concat, incols={'col_0':'x', 'col_2':'y'},
                         outcols={'out':str},
                         kwargs={}
                         )
    print(res.sample())
