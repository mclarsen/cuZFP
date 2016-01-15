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
#include "encode.cuh"
#include "decode.cuh"

using namespace thrust;
using namespace std;

#define FREXP(x, e) frexp(x, e)
#define LDEXP(x, e) ldexp(x, e)
#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const int nx = 256;
const int ny = 256;
const int nz = 256;
device_vector<long long> d_vec_buffer(nx*ny*nz);

static const unsigned char
perm[64] = {
  index(0, 0, 0), //  0 : 0

  index(1, 0, 0), //  1 : 1
  index(0, 1, 0), //  2 : 1
  index(0, 0, 1), //  3 : 1

  index(0, 1, 1), //  4 : 2
  index(1, 0, 1), //  5 : 2
  index(1, 1, 0), //  6 : 2

  index(2, 0, 0), //  7 : 2
  index(0, 2, 0), //  8 : 2
  index(0, 0, 2), //  9 : 2

  index(1, 1, 1), // 10 : 3

  index(2, 1, 0), // 11 : 3
  index(2, 0, 1), // 12 : 3
  index(0, 2, 1), // 13 : 3
  index(1, 2, 0), // 14 : 3
  index(1, 0, 2), // 15 : 3
  index(0, 1, 2), // 16 : 3

  index(3, 0, 0), // 17 : 3
  index(0, 3, 0), // 18 : 3
  index(0, 0, 3), // 19 : 3

  index(2, 1, 1), // 20 : 4
  index(1, 2, 1), // 21 : 4
  index(1, 1, 2), // 22 : 4

  index(0, 2, 2), // 23 : 4
  index(2, 0, 2), // 24 : 4
  index(2, 2, 0), // 25 : 4

  index(3, 1, 0), // 26 : 4
  index(3, 0, 1), // 27 : 4
  index(0, 3, 1), // 28 : 4
  index(1, 3, 0), // 29 : 4
  index(1, 0, 3), // 30 : 4
  index(0, 1, 3), // 31 : 4

  index(1, 2, 2), // 32 : 5
  index(2, 1, 2), // 33 : 5
  index(2, 2, 1), // 34 : 5

  index(3, 1, 1), // 35 : 5
  index(1, 3, 1), // 36 : 5
  index(1, 1, 3), // 37 : 5

  index(3, 2, 0), // 38 : 5
  index(3, 0, 2), // 39 : 5
  index(0, 3, 2), // 40 : 5
  index(2, 3, 0), // 41 : 5
  index(2, 0, 3), // 42 : 5
  index(0, 2, 3), // 43 : 5

  index(2, 2, 2), // 44 : 6

  index(3, 2, 1), // 45 : 6
  index(3, 1, 2), // 46 : 6
  index(1, 3, 2), // 47 : 6
  index(2, 3, 1), // 48 : 6
  index(2, 1, 3), // 49 : 6
  index(1, 2, 3), // 50 : 6

  index(0, 3, 3), // 51 : 6
  index(3, 0, 3), // 52 : 6
  index(3, 3, 0), // 53 : 6

  index(3, 2, 2), // 54 : 7
  index(2, 3, 2), // 55 : 7
  index(2, 2, 3), // 56 : 7

  index(1, 3, 3), // 57 : 7
  index(3, 1, 3), // 58 : 7
  index(3, 3, 1), // 59 : 7

  index(2, 3, 3), // 60 : 8
  index(3, 2, 3), // 61 : 8
  index(3, 3, 2), // 62 : 8

  index(3, 3, 3), // 63 : 9
};



void setupConst(const unsigned char *perm)
{
    ErrorCheck ec;
    ec.chk("setupConst start");
    cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char)*64,0); ec.chk("setupConst: lic_dim");
    ec.chk("setupConst finished");


}



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
void cpuTestDecorrelate
(
        Scalar *p
        )
{
//#pragma omp parallel for
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

template<class Int, class UInt>
void reorder
(
        const Int *q,
        UInt *buffer
        )
{
    for (uint i = 0; i < 64; i++)
      buffer[i] = int2uint<Int, UInt>(q[perm[i]]);
}


template<class Int, class UInt, class Scalar>
void gpuTestinv_xform
(
        device_vector<Int> &q
        )
{
    device_vector<Int> q_out;
    q_out.resize(nx*ny*nz);
    q_out = q;
    dim3 emax_size(nx/4, ny/4, nz/4 );

    dim3 block_size(8,8,8);
    dim3 grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

    ErrorCheck ec;

    block_size = dim3(8,8,8);
    grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

    cudaInvXForm<Int><<<block_size, grid_size>>>
        (
            raw_pointer_cast(q_out.data())
        );
    cudaStreamSynchronize(0);
    ec.chk("cudaInvXForm");


    host_vector<Int> h_q, h_qout;
    h_q = q;
    h_qout = q_out;

    int i=0;
    for (int z=0; z<nz; z+=4){
        for (int y=0; y<ny; y+=4){
            for (int x=0; x<nx; x+=4){
                int idx = z*nx*ny + y*nx + x;
                Int iblock[64];
                for (int j=0; j<64; j++)
                    iblock[j] = h_q[i*64 + j];
                inv_xform(iblock);

                for (int j=0; j<64;j++){
                    assert(iblock[j] == h_qout[i*64 + j]);
                }

                i++;

            }
        }
    }
}

int main()
{
    host_vector<long long> h_q;
    h_q.resize(nx*ny*nz);
    for (int i=0; i<h_q.size(); i++){
        h_q[i] = i;
    }

    d_vec_buffer = h_q;

    setupConst(perm);
    gpuTestinv_xform<long long, unsigned long long, double>(d_vec_buffer);
}
