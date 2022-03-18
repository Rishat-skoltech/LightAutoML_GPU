#include "dev_bucket_sort.hh"

template <typename T>
__global__ void find_nans(T * d_vector, uint length, uint * mask, uint nans_flag)
{
    int idx = threadIdx.x+blockIdx.x*blockDim.x;
    int offset = blockDim.x*gridDim.x;
    for (int i = idx; i < length; i+=offset)
        if (isnan(d_vector[i]))
            mask[i] = nans_flag;
    return;
}

template <typename T>
__global__ void fill_nans(uint * mask, uint length, T* val, T * d_vector, uint nans_flag)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    for (int i = idx; i < length; i+=offset)
        if (mask[i]==nans_flag)
    	    d_vector[i] = val[0];
    return;
}

template <typename T>
__global__ void find_minmax(T * d_vector, T * d_minmax, uint length)
{
    int idx = threadIdx.x+blockIdx.x*2*blockDim.x;
    int offset = 2*blockDim.x*gridDim.x;
    T lhs, rhs, lhs2, rhs2;
    extern __shared__ T sdata[];
    int counter = 0;
    for (int i = idx; i < length; i+=offset)
    {
        sdata[threadIdx.x] = d_vector[i];
        sdata[threadIdx.x+blockDim.x] = d_vector[i+blockDim.x];
        __syncthreads();
        for (int stride = blockDim.x; stride > 0; stride /=2)
        {
            if(threadIdx.x < stride)
            {
                lhs = sdata[threadIdx.x];
                rhs = sdata[threadIdx.x+stride];
            }
            if (blockDim.x - threadIdx.x-1 < stride)
            {
                lhs2 = sdata[threadIdx.x+blockDim.x];
                rhs2 = sdata[threadIdx.x+blockDim.x-stride];
            }
            __syncthreads();
            if (threadIdx.x < stride)
                sdata[threadIdx.x] = lhs = isnan(lhs) ? rhs : (lhs<rhs ? rhs : lhs);                 
            if (blockDim.x - threadIdx.x-1 < stride)
                sdata[threadIdx.x+blockDim.x] = isnan(lhs2) ? rhs2 : (lhs2>rhs2 ? rhs2 : lhs2);
            __syncthreads();
        }
        if (threadIdx.x == 0)
            d_minmax[blockIdx.x + counter*gridDim.x] = sdata[0];
        if (threadIdx.x == blockDim.x - 1)
            d_minmax[blockIdx.x+gridDim.x + counter*gridDim.x] = sdata[2*blockDim.x-1];
        counter++;
    }
    return;
}

template <typename T>
__global__ void assignBucket(T * d_vector, uint length, uint bucketNumbers, T slope, T sMin, uint * bucket, uint * bucketCount, uint * mask, uint nan_flag)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;
  int bucketIndex;
  extern __shared__ uint sharedBuckets[];
  for (int i = threadIdx.x; i < bucketNumbers; i+=blockDim.x)
      sharedBuckets[i] = 0;
  __syncthreads();
  for (int i = idx; i < length; i+=offset)
  {
      if (mask[i]!=nan_flag)
      {
          bucketIndex = (d_vector[i] - sMin) * slope;
          if (bucketIndex >= bucketNumbers)
              bucketIndex = bucketNumbers - 1;
          bucket[i] = bucketIndex;
          atomicInc(&sharedBuckets[bucketIndex], length);
      }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < bucketNumbers; i+=blockDim.x)
      atomicAdd(&bucketCount[i], sharedBuckets[i]);
}

template <typename T>
__global__ void reassignBucket(T * d_vector, uint * bucket, uint * bucketCount, const uint bucketNumbers, const uint length, const T slope, T minimum, uint Kbucket, uint nan_flag) {
  int idx    = blockDim.x * blockIdx.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  extern __shared__ uint sharedBuckets[];
  int bucketIndex;
  for (int i = threadIdx.x; i < bucketNumbers; i+=blockDim.x)
      sharedBuckets[i] = 0;
  __syncthreads();
  // assigning elements to buckets and incrementing the bucket counts
    for (int i = idx; i < length; i += stride) {
      if (bucket[i] == nan_flag){}
      else if (bucket[i] != Kbucket) {
        bucket[i] = bucketNumbers + 1;
      } 
      else {
        // calculate the bucketIndex for each element
        bucketIndex = (d_vector[i] - minimum) * slope;

        // if it goes beyond the number of buckets, put it in the last bucket
        if (bucketIndex >= bucketNumbers) { bucketIndex = bucketNumbers - 1; }
        bucket[i] = bucketIndex;
        atomicInc(&sharedBuckets[bucketIndex], length);
      }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < bucketNumbers; i+=blockDim.x)
      atomicAdd(&bucketCount[i], sharedBuckets[i]);
}

inline int FindKBucket(uint *d_counter, const int numBuckets, const int k, uint *sum)
{
  int Kbucket = 0;
  if (*sum < k) {
    while ((*sum < k) & (Kbucket < numBuckets - 1)) {
      Kbucket++;
      *sum += d_counter[Kbucket];
    }
  }

  return Kbucket;
}

template <typename T>
__global__ void GetKvalue(T *d_vector, uint *d_bucket, const uint Kbucket, const uint n, T *Kvalue) {
  int idx    = blockDim.x * blockIdx.x + threadIdx.x;
  int offset = blockDim.x * gridDim.x;

  for (int i = idx; i < n; i += offset)
      if (d_bucket[i] == Kbucket) Kvalue[0] = d_vector[i];
}

template <typename T>
T * phaseTwo(T *d_vector, uint *mask, uint length, uint K, uint blocks, uint threads,  T * d_minmax, uint nan_flag)
{
    T minValue = d_minmax[1];
    T maxValue = d_minmax[0];
    if(minValue == maxValue)
    {
        return d_minmax;
    }	
    int numBuckets = length < 1024 ? length+1 : 1024;
    uint sum = 0, Kbucket = 0, iter = 0;
    int  Kbucket_count = 0;

    uint *d_bucketCount;
    cudaMallocManaged(&d_bucketCount, numBuckets * sizeof(uint));
    cudaMemset(d_bucketCount, 0, numBuckets * sizeof(uint));
    T *d_Kth_val;
    cudaMallocManaged(&d_Kth_val, sizeof(T));
  
    T slope = (numBuckets - 1) / (maxValue - minValue);
    assignBucket<<<blocks, threads, numBuckets * sizeof(uint)>>>(d_vector, length, numBuckets, slope, minValue, mask, d_bucketCount, mask, nan_flag);
    cudaDeviceSynchronize();

    Kbucket = FindKBucket(d_bucketCount, numBuckets, K, &sum);
    Kbucket_count = d_bucketCount[Kbucket];

    while ((Kbucket_count > 1) && (iter < 1000))
    {
        minValue = max(minValue, minValue + Kbucket / slope);
        maxValue = min(maxValue, minValue + 1 / slope);

        K = K - sum + Kbucket_count;

        if (maxValue - minValue > 0.0f)
        {
            slope = (numBuckets - 1) / (maxValue - minValue);
            cudaMemset(d_bucketCount, 0, numBuckets * sizeof(uint));
            reassignBucket<<<blocks, threads, numBuckets * sizeof(uint)>>>(d_vector, mask, d_bucketCount, numBuckets, length, slope, minValue, Kbucket, nan_flag);
            cudaDeviceSynchronize();
            sum = 0;
            Kbucket = FindKBucket(d_bucketCount, numBuckets, K, &sum);
            Kbucket_count = d_bucketCount[Kbucket];

            iter++;
        } 
        else
        {
            d_Kth_val[0] = maxValue;
            return d_Kth_val;
        }
    }
    GetKvalue<<<blocks, threads>>>(d_vector, mask, Kbucket, length, d_Kth_val);
    cudaDeviceSynchronize();
    return d_Kth_val;
}

void fill_na_median_d(double * d_v, uint N)
{
    uint K = N/2;
    int threads = (N+1)/2 > 1024 ? 1024 : (N+1)/2;
    int blocks = ((N+1)/2 + threads - 1)/threads;
    
    uint * d_nans;
    cudaMallocManaged(&d_nans, sizeof(uint)*N);
    double * d_minmax;
    cudaMallocManaged(&d_minmax, sizeof(double)*2*blocks);

    uint NAN_FLAG = N+3;
    find_nans<<<blocks, threads*2>>>(d_v, N, d_nans, NAN_FLAG);

    find_minmax<<<blocks, threads, 2*threads*sizeof(double)>>>(d_v, d_minmax, N);
    if(blocks > 1)
    	find_minmax<<<1, blocks, 2*blocks*sizeof(double)>>>(d_minmax, d_minmax, 2*blocks);

    double * d_median;
    cudaMallocManaged(&d_median, sizeof(double));
    cudaDeviceSynchronize();
    int new_blocks = threads >= 1024 ? 2*blocks : blocks;
    int new_threads = threads >= 1024 ? 1024 : 2*threads;
    d_median = phaseTwo(d_v, d_nans, N, K, new_blocks, new_threads,  d_minmax, NAN_FLAG);
    fill_nans<<<blocks, threads*2>>>(d_nans, N, &d_median[0], d_v, NAN_FLAG);
    cudaDeviceSynchronize();
    cudaFree(d_nans);
    cudaFree(d_minmax);
    cudaFree(d_median);
    cudaDeviceSynchronize();
    return;
}

/*int main (int argc, char ** argv)
{
    uint N = atoi(argv[1]);
    std::srand(unsigned(std::time(nullptr)));
    std::vector<double> v(N);
    std::generate(v.begin(), v.end(), [n=1] () mutable { return  (double)n++; } );
    v[2] = std::nan("1");
    v[6] = std::nan("1");
    v[14] = std::nan("1");
    double * d_v;
    cudaMallocManaged((void**)&d_v, N*sizeof(double));
    cudaMemcpy(d_v, v.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    fill_na_median_d(d_v, N);
    cudaFree(d_v);
    return 0;
}*/
