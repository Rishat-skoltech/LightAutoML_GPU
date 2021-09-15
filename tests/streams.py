import numpy as np
import cupy as cp
import sys
from time import perf_counter
from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Process

def gpu_func(num, a, b):
    res = 0
    a_cur = a
    b_cur = b
    for i in range(100):
        c = a_cur-b_cur
        res +=c.sum()
    two = np.random.random(100)
    #print(res, "GPU")
    return res, two[0]

def cpu_func(a_n, b_n):   
    res_n = 0
    for i in range(500):
        c_n = a_n/b_n
        res_n += c_n.sum()
    print(res_n, "CPU")
    return res_n
    
#print(cp.cuda.runtime.getDevice())
def main():
    
    map_streams = []
    res = []
    for i in range(8):
        map_streams.append(cp.cuda.stream.Stream())

    st = perf_counter()
    #for stream in map_streams:
        #with stream:
        #    a = cp.random.random(10000000)
        #    b = cp.random.random(10000000)
        #    g = gpu_func(i, a, b)
        #    res.append(g)
    for i in range(len(map_streams)):
        a = cp.random.random(10000000)
        b = cp.random.random(10000000)
        g = gpu_func(i, a, b)
        res.append(g)
    #cp.cuda.Device().synchronize()
    print(perf_counter() - st, "TOTAL TIME")
    print(res)
    sys.exit()

if __name__ == "__main__":
    main()
