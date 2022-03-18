from numba import jit, cuda
import numpy as np
import pandas as pd
from memory_profiler import profile
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

#@profile
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
