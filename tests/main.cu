#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

using namespace thrust;
using namespace std;

#define FREXP(x, e) frexp(x, e)
#define LDEXP(x, e) ldexp(x, e)

const int nx = 32;
const int ny = 32;
const int nz = 32;
uint mx = 0;
uint my = 0;
uint mz = 0;
size_t blksize = 0;

struct RandGen
{
    RandGen() {}

    __device__ float operator () (const uint idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist(-1.0, 1.0);
        randEng.discard(idx);
        return uniDist(randEng);
    }
};

template<class T>
__global__
void testFREXP
(
        int max_threads,
        const T *in,
        T *out
        )
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < max_threads)
        out[idx] = LDEXP(in[idx], 10);

}

template<class T>
__global__
void testLDEXP(
        int max_threads,
        const T *in,
        T *out
        )
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < max_threads)
        out[idx] = LDEXP(in[idx], 10);
}


int main()
{
    device_vector<double> d_vec_in(nx*ny*nz), d_vec_out(nx*ny*nz);
    host_vector<double> h_vec(nx*ny*nz);

    thrust::counting_iterator<uint> index_sequence_begin(0);
    thrust::transform(
                    index_sequence_begin,
                    index_sequence_begin + nx*ny*nz,
                    d_vec_in.begin(),
                    RandGen());

    double *raw_in = raw_pointer_cast(d_vec_in.data());
    double *raw_out = raw_pointer_cast(d_vec_out.data());

    testLDEXP<double><<<nx*ny, nz>>>(
        nx*ny*nz,
        raw_in,
        raw_out
    );
    double sum = reduce(
        d_vec_in.begin(),
        d_vec_in.end()
    );

    cout << "LDEXP sum: " << sum << endl;
    h_vec = d_vec_out;
}
