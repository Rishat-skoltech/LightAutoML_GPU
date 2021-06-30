cdef extern from "dev_bucket_sort.hh":
    void fill_na_median_d(double * d_v, size_t N)
    
cdef fillnamedian_wrapper(size_t d_v, size_t N):
    fill_na_median_d(<double*>d_v, N)

def fillnamedian(d_v, size_t N):
    vPtr = d_v.data.ptr
    fillnamedian_wrapper(vPtr, N)
