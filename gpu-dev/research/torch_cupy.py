from memory_profiler import profile
import cupy as cp
import torch


@profile
def torch_to_cupy(data):
    return cp.asarray(data)

@profile
def cupy_to_torch(data):
    return torch.as_tensor(data, device='cuda')
    
@profile
def main():
    N = 10**8
    cupy_array = cp.arange(N)
    torch_array = torch.rand(N, device='cuda')
    
    cupy_array_from_torch = torch_to_cupy(torch_array)
    torch_array_from_cupy = cupy_to_torch(cupy_array)
    
    
if __name__ == "__main__":
    main()
