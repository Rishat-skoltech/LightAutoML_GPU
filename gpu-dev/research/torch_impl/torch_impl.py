import torch
from memory_profiler import profile
import copy

#@profile
def FillnaMedian_torch(data):
    output = copy.deepcopy(data)
    median = torch.nanmedian(data, dim=1)
    for i in range(output.shape[0]):
        output[i] = torch.nan_to_num(output[i], nan=median[0][i])
    return output
