#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <assert.h>

#define KEPLER 0
#include "ErrorCheck.h"
#include "fixed_point.cuh"

using namespace thrust;
using namespace std;

#define FREXP(x, e) frexp(x, e)
#define LDEXP(x, e) ldexp(x, e)

const int nx = 256;
const int ny = 256;
const int nz = 256;
device_vector<double> d_vec_in(nx*ny*nz);
device_vector<long long> d_vec_out(nx*ny*nz);
host_vector<double> h_vec_in(nx*ny*nz);


//Used to generate rand array in CUDA with Thrust
struct RandGen
{
    RandGen() {}

    __device__ float operator () (const uint idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist(0.0, 0.0001);
        randEng.discard(idx);
        return uniDist(randEng);
    }
};

template<class Int, class Scalar>
void cpuTestFixedPoint
(
        Scalar *p
        )
{
#pragma omp parallel for
    for (int z=0; z<nz; z+=4){
        for (int y=0; y<ny; y+=4){
            for (int x=0; x<nx; x+=4){
                int idx = z*nx*ny + y*nx + x;
                Int q[64];
                Int q2[64];
                int emax2 = max_exp<Scalar>(p, idx, 1,nx,nx*ny);
                fixed_point(q2,p, emax2, idx, 1,nx,nx*ny);

                int emax = fwd_cast(q, p+idx, 1,nx,nx*ny);

                for (int i=0; i<64; i++){
                    assert(q[i] == q2[i]);
                }

            }
        }
    }

}
template<class Int, class Scalar>
__global__
void cudaFixedPoint
(
        const int *emax,
        const Scalar *data,
        Int *q
        )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int eidx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;

    x *= 4; y*=4; z*=4;
    int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
    fixed_point(q, data, emax[eidx], idx, gridDim.x*blockDim.x*4, gridDim.y*blockDim.y*4, gridDim.z*blockDim.z*4);
}

template<class Scalar>
__global__
void cudaMaxExp
(
        int *emax,
    Scalar *data
        )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int eidx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;

    x *= 4; y*=4; z*=4;
    int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
    emax[eidx] = max_exp(data, idx, gridDim.x*blockDim.x*4, gridDim.y*blockDim.y*4, gridDim.z*blockDim.z*4);

}

template<class Int, class Scalar>
void gpuTestFixedPoint
(
        device_vector<Scalar> &data,
        device_vector<Int> &q,
        device_vector<int> &emax
        )
{
    dim3 emax_size(nx/4, ny/4, nz/4 );

    dim3 block_size(8,8,8);
    dim3 grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

    ErrorCheck ec;

    ec.chk("pre-cudaMaxExp");
    cudaMaxExp<<<block_size,grid_size>>>
            (
                raw_pointer_cast(emax.data()),
                raw_pointer_cast(data.data())
                );
    ec.chk("cudaMaxExp");

    ec.chk("pre-cudaFixedPoint");
    cudaFixedPoint<<<block_size, grid_size>>>
            (
                raw_pointer_cast(emax.data()),
                raw_pointer_cast(data.data()),
                raw_pointer_cast(q.data())
                );
    ec.chk("cudaFixedPoint");
    host_vector<int> h_emax;
    host_vector<Scalar> h_p;
    host_vector<Int> h_q;
    h_emax = emax;
    h_p = data;
    h_q = q;

    int i=0;
    for (int z=0; z<nz; z+=4){
        for (int y=0; y<ny; y+=4){
            for (int x=0; x<nx; x+=4){
                int idx = z*nx*ny + y*nx + x;
                Int q2[64];
                int emax2 = max_exp<Scalar>(raw_pointer_cast(h_p.data()), idx, 1,nx,nx*ny);
                assert(emax2 == h_emax[i++]);
                fixed_point(q2,raw_pointer_cast(h_p.data()), emax2, idx, 1,nx,nx*ny);
                for (int j=0; j<64; j++){
                    assert(h_q[j+(i-1)*64] == q2[j]);
                }
            }
        }
    }
}

int main()
{

    dim3 emax_size(nx/4, ny/4, nz/4);
    device_vector<int> emax(emax_size.x * emax_size.y * emax_size.z);

    thrust::counting_iterator<uint> index_sequence_begin(0);
    thrust::transform(
                    index_sequence_begin,
                    index_sequence_begin + nx*ny*nz,
                    d_vec_in.begin(),
                    RandGen());

    gpuTestFixedPoint<long long, double>(d_vec_in, d_vec_out, emax);
    h_vec_in = d_vec_in;
    cpuTestFixedPoint<long long, double>(raw_pointer_cast(h_vec_in.data()));
}
