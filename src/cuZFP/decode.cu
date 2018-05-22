#include "decode.cuh"
#include "cuZFP.h"

#include <vector>

namespace cuZFP {

template void decode<double>
            (int3 dims, 
             Word *stream,
             double *d_data,
             uint bsize);

template void decode<float>
            (int3 dims, 
             Word *stream,
             float *d_data,
             uint bsize);

template void decode<long long int>
            (int3 dims, 
             Word *stream,
             long long int *d_data,
             uint bsize);

template void decode<int>
            (int3 dims, 
             Word *stream,
             int *d_data,
             uint bsize);
}
