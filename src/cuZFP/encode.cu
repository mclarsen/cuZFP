#include "encode.cuh"

namespace cuZFP {
template void encode<long long int, unsigned long long int, double, 8, 64>
(
int nx, int ny, int nz,
const thrust::host_vector<double> &h_data,
thrust::host_vector<Word> &stream,
const unsigned long long group_count,
const uint size
);

}
