cimport numpy as np

cdef extern from "dev_bucket_sort.hh":
    void fill_na_median_cpu(double * d_v, size_t N)
    
def fillnamedian_cpu(np.ndarray[ndim=1, dtype=np.float64_t] d_v, N):
    fill_na_median_cpu(&d_v[0], N)
