
from lightautoml.dataset.cp_cudf_dataset import *
from lightautoml.dataset.daskcudf_dataset import *

from lightautoml.dataset.roles import *
from lightautoml.dataset.utils import roles_parser

from lightautoml.tasks import Task

from lightautoml.validation.gpu_iterators import FoldsIterator_gpu

from lightautoml.reader.daskcudf_reader import DaskCudfReader

from time import perf_counter

import numpy as np
import random

from numba import jit
import string

from lightautoml.ml_algo.linear_gpu import LinearLBFGS_gpu

RANDS_CHARS = np.array(list(string.ascii_letters + string.digits),
                       dtype=(np.str_, 1))


@jit(nopython=True)
def gen_cols(n_cols):
    cols = [""]*n_cols
    for i in range(n_cols):
        cols[i] = "col_" + str(i)
    return cols

def gen_string_data(n, n_str):
    string_db = ["algorithm", "analog", "app", "application", "array",
                 "backup", "bandwidth", "binary", "bit", "byte"]#,
                 #"bitmap", "blog", "bookmark", "boot", "broadband",
                 #"browser" , "buffer", "bug"]
    inds = np.random.randint(0, len(string_db), (n, n_str))
    output = np.empty(inds.shape, dtype=object)
    for i in range(inds.shape[0]):
        for j in range(inds.shape[1]):
            output[i][j] = string_db[inds[i][j]]

    return output

def generate_data(n, n_num, n_cat, n_date, n_str, max_n_cat):
    print("Generating dummy data")
    n_cols = n_num+n_cat+n_str+n_date
    cols = gen_cols(n_cols)
    data = np.random.random((n, n_num))*100-50

    category_data = np.random.randint(0, np.random.randint(1,max_n_cat), (n, n_cat))
    string_data = gen_string_data(n, n_str)

    string_data = np.reshape(string_data, (n, n_str))

    date_data = np.random.randint(0, 1000,
                               (n, n_date)).astype(np.dtype("timedelta64[D]")) \
                               + np.datetime64("2018-01-01")

    data = pd.DataFrame(data, columns = cols[:n_num]).astype('f')
    
    ix = [(row, col) for row in range(data.shape[0]) for col in range(data.shape[1])]
    for row, col in random.sample(ix, int(round(.0*len(ix)))):
        data.iat[row, col] = np.nan
    
    nn = len(data.columns)
    for i in range(n_cat):
        data[cols[nn+i]] = pd.Series(category_data[:,i]).astype('f')
    nn = len(data.columns)
    for i in range(n_str):
        data[cols[nn+i]] = pd.Series(string_data[:,i]).astype(object)
    nn = len(data.columns)
    for i in range(n_date):
        data[cols[nn+i]] = pd.Series(date_data[:,i])

    data['col_m'] = pd.Series(np.zeros(n))
    data['col_n'] = pd.Series(np.zeros(n))
    data['target'] = pd.Series(np.random.randint(0, 2, n)).astype('i')

    print("Shape of the dummy data:", data.shape)
    print("Size of the dummy data:",
          round(data.memory_usage(deep=True).sum()/1024./1024.,4), "MB.")
    return 'target', cols, data

def main():
    task = Task("binary", device="mgpu")

    target, _, data = generate_data(n=40, n_num=4, n_cat=0, n_date=0,
                                    n_str=0, max_n_cat=10)

    daskcudf_reader = DaskCudfReader(task, device_num=0, cv=5, random_state=42, n_jobs=1, compute=True, npartitions=1, advanced_roles=True)

    start = perf_counter()
    ddf_dataset = daskcudf_reader.fit_read(data, target=data[target])
    print("DaskCudfReader fit_read time:", perf_counter()-start,
          "seconds, the shape of the data is", ddf_dataset.data.shape)

    #print(ddf_dataset.data.compute())

    train_valid = FoldsIterator_gpu(ddf_dataset)


    

    # print(train_valid.train.data.compute())
    linear = LinearLBFGS_gpu()

    linear.fit_predict(train_valid)

    print("Finished")


if __name__ == '__main__':
    main()


