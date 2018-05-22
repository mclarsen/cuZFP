#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"
#include "cuZFP.h"

#include <vector>

namespace cuZFP {

template size_t encode<double>
            (int3 dims,
             double *d_data,
             Word *stream,
             int bsize);

template size_t encode<float>
            (int3 dims,
             float *d_data,
             Word *stream,
             int bsize);

template size_t encode<long long int>
            (int3 dims,
             long long int *d_data,
             Word *stream,
             int bsize);

template size_t encode<int>
            (int3 dims,
             int *d_data,
             Word *stream,
             int bsize);

template size_t encode1<float>
            (int dim,
             float *d_data,
             Word *stream,
             int bsize);

template size_t encode2<float>
            (int2 dims,
             float *d_data,
             Word *stream,
             int bsize);

}
