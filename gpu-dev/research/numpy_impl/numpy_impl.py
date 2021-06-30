import numpy as np
from memory_profiler import profile

#@profile
def FillnaMedian_numpy(data):
    output = np.copy(data)
    median = np.nanmedian(data, axis=1)
    for i in range(output.shape[0]):
        output[i] = np.nan_to_num(output[i], nan=median[i])
    return output
