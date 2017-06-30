#include "decode.cuh"
#include "cuZFP.h"

#include <vector>

namespace cuZFP {

template void decode<double>
            (int3 dims, 
             thrust::device_vector<Word> &stream,
             thrust::device_vector<double> &d_data,
             uint bsize);

template void decode<float>
            (int3 dims, 
             thrust::device_vector<Word> &stream,
             thrust::device_vector<float> &d_data,
             uint bsize);
}
