import cupy  
import dask.array as da
import numpy as np

rs = da.random.RandomState(RandomState=cupy.random.RandomState)  
dt = np.float32

print(da.map_blocks(lambda x: x.argmax(axis=1), rs.random((5,2), dtype=dt), meta=cupy.array((), dtype=dt)).compute() )
