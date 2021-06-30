import fillnamedian_cpu
import numpy as np

N = 128
arr = np.arange(N, dtype=np.float64)

arr[10] = np.nan
arr[33] = np.nan
print(arr)
fillnamedian_cpu.fillnamedian_cpu(arr, N)
print(arr)
