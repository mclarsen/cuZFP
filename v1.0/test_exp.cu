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
#include "encode.cuh"

using namespace thrust;
using namespace std;


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
        thrust::uniform_real_distribution<float> uniDist;
        randEng.discard(idx);
        return uniDist(randEng);
    }
};



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
    if (idx < max_threads){
        setFREXP(idx, in, out, nptr);
    }

}



//*****************************************************************
//testLDEXP
//Input:
//max_threads, number of items in in and out arrays
//array of in, of type T
//Output: out array
//*****************************************************************
template<class T, bool mult_only>
__global__
void cudaTestLDEXP(
        int max_threads,
        const T *in,
        T *out,
        const T w,
        const int exp
        )
{
    uint idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < max_threads)
        setLDEXP<T, mult_only>(idx, in, out, w, exp);
}

template<class T>
void GPUTestFREXP(
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

    cout << "FREXP GPU sum: " << sum << " in time: " << millisecs << endl;

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

}

template<class T, bool mult_only>
void GPUTestLDEXP(
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

    int emax = 0;
    int exp = intprec -2 -emax;
    double w = LDEXP(1, exp);

    ec.chk("pre-testLDEXP");
    cudaTestLDEXP<T, mult_only><<<grid_size, block_size>>>(
        nx*ny*nz,
        raw_pointer_cast(in.data()),
        raw_pointer_cast(out.data()),
        w,
        exp
    ); ec.chk("testLDEXP");
    T sum = reduce(
                out.begin(),
                out.end()
    );
    cudaStreamSynchronize(0);

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &millisecs, start, stop );
    cout << "LDEXP GPU ";
    if (mult_only)
        cout << " multiplication shortcut ";
    else
        cout << "no shortcut ";
    cout << "sum: " << sum << " in time: " << millisecs << endl;

}

template<class T>
void CPUTestFREXP
(
        host_vector<T> &h_vec_in,
        host_vector<T> &h_vec_out,
        host_vector<int> &h_vec_nptr
        )
{
    std::fill(h_vec_out.begin(), h_vec_out.end(), 0);
    for (int i=0; i<nx*ny*nz; i++){
        setFREXP<T>
                (
                    i,
                    raw_pointer_cast( h_vec_in.data() ),
                    raw_pointer_cast(h_vec_out.data()),
                    raw_pointer_cast(h_vec_nptr.data())
                    );
    }
    cout << "FREXP CPU sum: " << reduce(h_vec_out.begin(),h_vec_out.end()) << endl;

}

template<class T, bool mult_only>
void CPUTestLDEXP
(
        host_vector<T> &h_vec_in,
        host_vector<T> &h_vec_out
        )
{
    const int intprec = 64;
    int emax = 0;
    int exp = intprec -2 -emax;
    double w = LDEXP(1, exp);

    for (int i=0; i<nx*ny*nz; i++){
        setLDEXP<T, mult_only>
                (
                    i,
                    raw_pointer_cast(h_vec_in.data()),
                    raw_pointer_cast(h_vec_out.data()),
                    w,
                    intprec
                 );
    }
    cout << "LDEXP CPU sum: " << reduce(h_vec_out.begin(), h_vec_out.end()) << endl;

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

    GPUTestFREXP<double>(d_vec_in, d_vec_out);
    GPUTestLDEXP<double, false>(d_vec_in, d_vec_out);
    GPUTestLDEXP<double, true>(d_vec_in, d_vec_out);

    host_vector<double> h_vec_out(nx*ny*nz);
    host_vector<int> h_vec_nptr(nx*ny*nz);

    h_vec_in = d_vec_in;
    CPUTestFREXP<double>(h_vec_in, h_vec_out, h_vec_nptr);
    CPUTestLDEXP<double, false> (h_vec_in, h_vec_out);
    CPUTestLDEXP<double, true> (h_vec_in, h_vec_out);
}
