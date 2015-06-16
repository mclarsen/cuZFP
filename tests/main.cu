#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

#define KEPLER 0
#include "ErrorCheck.h"

using namespace thrust;
using namespace std;

#define FREXP(x, e) frexp(x, e)
#define LDEXP(x, e) ldexp(x, e)

const int nx = 256;
const int ny = 256;
const int nz = 256;


//Used to generate rand array in CUDA with Thrust
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
__host__ __device__
void setFREXP
    (
        uint idx,
        const T *in,
        T *out,
        int *nptr
        )
{
    out[idx] = FREXP(in[idx], &nptr[ idx] );
}

//*****************************************************************
//testFREXP
//Input:
//max_threads, number of items in in and out arrays
//array of in, of type T
//Output: out array
//*****************************************************************
template<class T>
__global__
void cudaTestFREXP
(
        int max_threads,
        const T *in,
        T *out,
        int *nptr
        )
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < max_threads)
        setFREXP(idx, in, out, nptr);

}

template<class T>
__device__ __host__
void setLDEXP
(
    uint idx,
        const T *in,
        T *out
        )
{
    out[idx] = LDEXP(in[idx], 10);
}

//*****************************************************************
//testLDEXP
//Input:
//max_threads, number of items in in and out arrays
//array of in, of type T
//Output: out array
//*****************************************************************
template<class T>
__global__
void cudaTestLDEXP(
        int max_threads,
        const T *in,
        T *out
        )
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < max_threads)
        setLDEXP(idx, in, out);
}

template<class T>
void testFREXP(
        device_vector<T> &in,
        device_vector<T> &out)
{
    ErrorCheck ec;
    device_vector<int> d_vec_nptr(nx*ny*nz);
    cudaEvent_t start, stop;
    float millisecs;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord( start, 0 );
    //stupid laptop with max dim of grid size of 2^15
    const int block_size = 512;
    const int grid_size = nx*ny*nz / block_size;

    cudaTestFREXP<T><<<grid_size, block_size>>>(
        nx*ny*nz,
        raw_pointer_cast(in.data()),
        raw_pointer_cast(out.data()),
        raw_pointer_cast(d_vec_nptr.data())
    ); ec.chk("testFREXP");

    T sum = reduce(
            out.begin(),
            out.end()
        );
    cudaStreamSynchronize(0);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &millisecs, start, stop );

    cout << "FREXP GPU sum: " << sum << " in time: " << time << endl;

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

}

template<class T>
void testLDEXP(
        device_vector<T> &in,
        device_vector<T> &out)
{
    ErrorCheck ec;
    cudaEvent_t start, stop;
    float millisecs;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord( start, 0 );

    //stupid laptop with max dim of grid size of 2^15
    const int block_size = 512;
    const int grid_size = nx*ny*nz / block_size;
    ec.chk("pre-testLDEXP");
    cudaTestLDEXP<T><<<grid_size, block_size>>>(
        nx*ny*nz,
        raw_pointer_cast(in.data()),
        raw_pointer_cast(out.data())
    ); ec.chk("testLDEXP");
    T sum = reduce(
                out.begin(),
                out.end()
    );
    cudaStreamSynchronize(0);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &millisecs, start, stop );
    cout << "LDEXP GPU sum: " << sum << " in time: " << millisecs << endl;

}

int main()
{
    device_vector<double> d_vec_in(nx*ny*nz), d_vec_out(nx*ny*nz);
    host_vector<double> h_vec_in(nx*ny*nz);

    thrust::counting_iterator<uint> index_sequence_begin(0);
    thrust::transform(
                    index_sequence_begin,
                    index_sequence_begin + nx*ny*nz,
                    d_vec_in.begin(),
                    RandGen());

    testFREXP<double>(d_vec_in, d_vec_out);
    testLDEXP<double>(d_vec_in, d_vec_out);

    host_vector<double> h_vec_out(nx*ny*nz);
    host_vector<int> h_vec_nptr(nx*ny*nz);

    h_vec_in = d_vec_in;
    for (int i=0; i<nx*ny*nz; i++){
        setFREXP(i, raw_pointer_cast( h_vec_in.data() ),
                raw_pointer_cast(h_vec_out.data()),
                raw_pointer_cast(h_vec_nptr.data()));
    }
    cout << "FREXP CPU sum: " << reduce(h_vec_out.begin(),h_vec_out.end()) << endl;

    for (int i=0; i<nx*ny*nz; i++){
        setLDEXP
                (
                    i,
                    raw_pointer_cast(h_vec_in.data()),
                    raw_pointer_cast(h_vec_out.data())
                 );
    }
    cout << "LDEXP CPU sum: " << reduce(h_vec_out.begin(), h_vec_out.end()) << endl;
}
