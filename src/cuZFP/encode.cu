#include "encode.cuh"
#include "cuZFP.h"

#include <vector>

namespace cuZFP {

template void encode<double>
            (int3 dims,
             thrust::device_vector<double> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template void encode<float>
            (int3 dims,
             thrust::device_vector<float> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template void encode<long long int>
            (int3 dims,
             thrust::device_vector<long long int> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template void encode<int>
            (int3 dims,
             thrust::device_vector<int> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template void encode1<float>
            (int dim,
             thrust::device_vector<float> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);
}
