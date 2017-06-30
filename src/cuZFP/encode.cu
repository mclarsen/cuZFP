#include "encode.cuh"
#include "cuZFP.h"

#include <vector>

namespace cuZFP {
/*
template void encode<long long int, unsigned long long int, double, 8, 64>
  (int nx, 
   int ny, 
   int nz,
   const thrust::host_vector<double> &h_data,
   thrust::host_vector<Word> &stream,
   const uint size);


template void encode<long long int, unsigned long long int, float, 8, 32>
  (int nx, 
   int ny, 
   int nz,
   const thrust::host_vector<float> &h_data,
   thrust::host_vector<Word> &stream,
   const uint size);
*/

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

//
//  Integers are not working yet
//
/*
template void encode<long long int, unsigned long long int, int, 8, 64>
  (int nx, 
   int ny, 
   int nz,
   const thrust::host_vector<int> &h_data,
   thrust::host_vector<Word> &stream,
   const unsigned long long group_count,
   const uint size);
*/
}
