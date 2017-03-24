#include "decode.cuh"
#include "cuZFP.h"

#include <vector>

namespace cuZFP {

template void decode<long long int, unsigned long long int, double, 16, 64>
            (int nx, 
             int ny, 
             int nz,
             thrust::device_vector<Word> &stream,
             thrust::device_vector<double> &d_data);
}
