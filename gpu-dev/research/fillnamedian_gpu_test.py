import cupy as cp
import cudf
import numpy as np
import timeit
import torch
import copy

from utils import utils as uu

from rapids_impl import rapids_impl as rr
from cuda_impl import fillnamedian as cu
from cpu_impl import fillnamedian_cpu as cpu
from torch_impl import torch_impl as tt
from numpy_impl import numpy_impl as nn
##################################################################################################
##################################################################################################
device = torch.device("cuda:0")
N = 32
N_num = 4
N_cat = 0
N_str = 0
nan_amount = int(N*N_num*0.1)

TARGET_NAME, cols, data = uu.generate_data(N, N_num, N_cat, N_str, nan_amount)
##################################################################################################
##################################################################################################
data_rapids = cudf.from_pandas(data)
data_numpy = np.copy(data.to_numpy().T)
data_torch = torch.from_numpy(data_numpy).float().to(device)
##################################################################################################
##################################################################################################
'''
print("CUDF:")
print(data_rapids)
print("NUMPY:")
print(data_numpy.T)
print("TORCH:")
print(data_torch.T)
'''
##################################################################################################
##################################################################################################
t1 = timeit.default_timer()
output_rapids = rr.FillnaMedian_rapids(data_rapids)
t2 = timeit.default_timer()
#cython version implementations can only transform single columns
output_cuda = cp.asarray(data["col_0"].to_numpy())
cu.fillnamedian(output_cuda, N)
t3 = timeit.default_timer()
#cython version implementations can only transform single columns
#cuda version gives a slightly shifted median
output_cpu = np.copy(data["col_0"].to_numpy())
cpu.fillnamedian_cpu(output_cpu, N)
t4 = timeit.default_timer()
output_numpy = nn.FillnaMedian_numpy(data_numpy)
t5 = timeit.default_timer()
output_torch = tt.FillnaMedian_torch(data_torch)
t6 = timeit.default_timer()
##################################################################################################
##################################################################################################
print("Wall time RAPIDS:", t2-t1, "seconds.")
print("Wall time custom cuda:", t3-t2, "seconds.")
print("Wall time custom cpu:", t4-t3, "seconds.")
print("Wall time NUMPY:", t5-t4, "seconds.")
print("Wall time TORCH:", t6-t5, "seconds.")
##################################################################################################
##################################################################################################

if np.allclose(cp.asnumpy(output_rapids.values).T, output_numpy):
    print("RAPIDS AND NUMPY OUTPUTS ARE THE SAME")
if np.allclose(output_numpy, cp.asnumpy(cp.asarray(output_torch))):
    print("NUMPY AND TORCH OUTPUTS ARE THE SAME")

print("CUDF:")
print(output_rapids)
print("NUMPY:")
print(output_numpy.T)
print("TORCH:")
print(output_torch.T)
print("CPU:")
print(output_cpu)
print("CUDA:")
print(output_cuda)

