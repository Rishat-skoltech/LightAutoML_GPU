import fillnamedian
import numpy as np
import cupy as cp

N = 128
arr = np.arange(N, dtype=np.float64)

arr[10] = np.nan
arr[33] = np.nan
print(arr)

cupy_arr = cp.asarray(arr)
fillnamedian.fillnamedian(cupy_arr, N)

res_arr = cp.asnumpy(cupy_arr)
print(res_arr)
