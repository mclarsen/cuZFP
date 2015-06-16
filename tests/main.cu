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
        out[idx] = FREXP(in[idx], &nptr[ idx] );

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
        out[idx] = LDEXP(in[idx], 10);
}

template<class T>
void testFEXP(
        device_vector<T> &in,
        device_vector<T> &out)
{
    device_vector<int> d_vec_nptr(nx*ny*nz);
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord( start, 0 );
    cudaTestFREXP<T><<<nx*ny, nz>>>(
        nx*ny*nz,
        raw_pointer_cast(in.data()),
        raw_pointer_cast(out.data()),
        raw_pointer_cast(d_vec_nptr.data())
    );
    T sum = reduce(
            out.begin(),
            out.end()
        );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &time, start, stop );

    cout << "FREXP sum: " << sum << " in time: " << time << endl;

    cudaEventDestroy( start );
    cudaEventDestroy( stop );

}

template<class T>
void testLDEXP(
        device_vector<T> &in,
        device_vector<T> &out)
{
    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord( start, 0 );

    cudaTestLDEXP<T><<<nx*ny, nz>>>(
        nx*ny*nz,
        raw_pointer_cast(in.data()),
        raw_pointer_cast(out.data())
    );
    T sum = reduce(
                out.begin(),
                out.end()
    );
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &time, start, stop );
    cout << "LDEXP sum: " << sum << " in time: " << time << endl;

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

    testFEXP<double>(d_vec_in, d_vec_out);
    testLDEXP<double>(d_vec_in, d_vec_out);

}
