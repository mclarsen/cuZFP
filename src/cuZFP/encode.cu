#include "encode.cuh"
#include "encode1.cuh"
#include "encode2.cuh"
#include "cuZFP.h"

#include <vector>

namespace cuZFP {

template size_t encode<double>
            (int3 dims,
             thrust::device_vector<double> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template size_t encode<float>
            (int3 dims,
             thrust::device_vector<float> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template size_t encode<long long int>
            (int3 dims,
             thrust::device_vector<long long int> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template size_t encode<int>
            (int3 dims,
             thrust::device_vector<int> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template size_t encode1<float>
            (int dim,
             thrust::device_vector<float> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

template size_t encode2<float>
            (int2 dims,
             thrust::device_vector<float> &d_data,
             thrust::device_vector<Word> &stream,
             int bsize);

}
