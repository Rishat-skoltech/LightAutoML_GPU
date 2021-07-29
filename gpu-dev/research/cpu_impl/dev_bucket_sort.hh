#include <stdio.h>
#include <vector>
#include <algorithm>
#include <ctime>
#include <experimental/random>
#include <iostream>
#include <omp.h>

using namespace std;

void fill_na_median_cpu(double * d_v, uint N);
