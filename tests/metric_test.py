import dask_cudf
import cudf
import pandas as pd
import numpy as np
import cupy as cp
from functools import partial

from lightautoml.tasks.common_metric_gpu import mean_absolute_error_mgpu,\
                 log_loss_mgpu, r2_score_mgpu, roc_auc_score_mgpu

n = 30
i = np.random.randint(0,4,n)
x = np.random.random((n))
y = np.random.random((n))
z = np.random.random((n,4))

x = pd.Series(x)
z = pd.DataFrame(z)
y = pd.Series(y)
i = pd.Series(i)

z_gpu = cudf.from_pandas(z)
x_gpu = cudf.from_pandas(x)
y_gpu = cudf.from_pandas(y)
i_gpu = cudf.from_pandas(i)

z_mgpu = dask_cudf.from_cudf(z_gpu, npartitions=3)
x_mgpu = dask_cudf.from_cudf(x_gpu, npartitions=3)
y_mgpu = dask_cudf.from_cudf(y_gpu, npartitions=3)
i_mgpu = dask_cudf.from_cudf(i_gpu, npartitions=3)

z_f = z_mgpu.to_dask_array(lengths=True, meta=cp.array((),dtype=cp.float32)).persist()
x_f = x_mgpu.to_dask_array(lengths=True, meta=cp.array((),dtype=cp.float32)).persist()
y_f = y_mgpu.to_dask_array(lengths=True, meta=cp.array((),dtype=cp.float32)).persist()
i_f = i_mgpu.to_dask_array(lengths=True, meta=cp.array((),dtype=cp.float32)).persist()

print(mean_absolute_error_mgpu(x_f, y_f))

print(r2_score_mgpu(x_f, y_f))

print(roc_auc_score_mgpu(i_f, y_f))

fu = partial(log_loss_mgpu, eps=1e-7) 
print(fu(i_f, z_f))

