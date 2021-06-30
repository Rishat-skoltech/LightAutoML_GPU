#include "dev_bucket_sort.hh"

void fill_na_median_cpu(double * h_v, uint N)
{
    double * temp = new double [N];
    #pragma omp paralel for
        for (uint i = 0; i < N; i++)
            temp[i] = h_v[i];
            
    uint K = N/2;
    std::nth_element(temp, &temp[K], &temp[N-1]);
    double median = temp[K];
    
    #pragma omp paralel for
    for (uint i = 0; i < N; i++)
        if(std::isnan(h_v[i]))
            h_v[i] = median;

    delete [] temp;
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
    fill_na_median_cpu(v.data(), N);
    for (uint i = 0; i < N; i++) std::cout << v[i] << " ";
    std::cout << std::endl;
    return 0;
}*/
