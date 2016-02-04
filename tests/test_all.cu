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
#include <omp.h>

#define KEPLER 0
#include "ErrorCheck.h"
#include "include/encode.cuh"
#include "include/decode.cuh"

using namespace thrust;
using namespace std;

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const size_t nx = 256;
const size_t ny = 256;
const size_t nz = 256;

uint minbits = 4096;
uint maxbits = 4096;
uint maxprec = 64;
int minexp = -1074;
const double rate = 64;
size_t  blksize = 0;
unsigned long long group_count = 0x46acca631ull;
uint size = 64;


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


static size_t block_size(double rate) { return (lrint(64 * rate) + CHAR_BIT - 1) / CHAR_BIT; }


template<class Scalar>
void setupConst(const unsigned char *perm)
{
    ErrorCheck ec;
    ec.chk("setupConst start");
    cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char)*64,0); ec.chk("setupConst: c_perm");

    const uint sizeof_scalar = sizeof(Scalar);
    cudaMemcpyToSymbol(c_sizeof_scalar, &sizeof_scalar, sizeof(uint)); ec.chk("setupConst: c_sizeof_scalar");

    ec.chk("setupConst finished");


}


/* reorder unsigned coefficients and convert to signed integer */
template<class Int, class UInt>
__host__
static void
inv_order(const UInt* ublock, Int* iblock, const unsigned char* perm, uint n)
{
  do
    iblock[*perm++] = uint2int<UInt>(*ublock++);
  while (--n);
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

template<class Int>
void gpuDecorrelate
(
        device_vector<Int> &q
        )
{
    ErrorCheck ec;
    dim3 block_size, grid_size;
    uint tot_size = 0;

    block_size = dim3(8, 8, 8);
    grid_size = dim3(nx,ny,nz);
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    grid_size.z /= 4;

    cudaDecorrelateZY<Int><<< grid_size, block_size>>>
        (
            raw_pointer_cast(q.data())
            );
    cudaThreadSynchronize();
    ec.chk("cudaDecorrelateZY");
    tot_size = nx*ny*nz;

    tot_size /= 16;
    block_size = dim3(8, 8, 4);
    grid_size.x = sqrt(tot_size);
    grid_size.y = sqrt(tot_size);
    grid_size.z = 1;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;

    cudaDecorrelateXZ<Int><<< grid_size, block_size>>>
        (
            raw_pointer_cast(q.data())
            );
    cudaThreadSynchronize();
    ec.chk("cudaDecorrelateXZ");

    tot_size = nx*ny*nz;
    tot_size /= 64;
    block_size = dim3(8, 8, 16);
    grid_size.x = sqrt(tot_size);
    grid_size.y = sqrt(tot_size);
    grid_size.z = 1;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;

    cudaDecorrelateYX<Int><<< grid_size, block_size>>>
        (
            raw_pointer_cast(q.data())
            );
    cudaThreadSynchronize();
    ec.chk("cudaDecorrelateYX");

}

template<class Int>
void gpuInvXform
(
        device_vector<Int> &q
        )
{
    ErrorCheck ec;
    dim3 block_size, grid_size;
    uint tot_size = 0;

    tot_size = nx*ny*nz;
    tot_size /= 64;
    block_size = dim3(8, 8, 16);
    grid_size.x = sqrt(tot_size);
    grid_size.y = sqrt(tot_size);
    grid_size.z = 1;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;

    cout << grid_size.x << " " << grid_size.y << " " << grid_size.z << endl;
    cudaInvXFormYX<Int> << <grid_size, block_size >> >
        (
        raw_pointer_cast(q.data())
        );
    cudaStreamSynchronize(0);
    ec.chk("cudaInvXFormYX");

    tot_size = nx*ny*nz;

    tot_size /= 16;
    block_size = dim3(8, 8, 4);
    grid_size.x = sqrt(tot_size);
    grid_size.y = sqrt(tot_size);
    grid_size.z = 1;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;

    cout << grid_size.x << " " << grid_size.y << " " << grid_size.z << endl;
    cudaInvXFormXZ<Int> << <grid_size, block_size >> >
        (
        raw_pointer_cast(q.data())
        );
    cudaStreamSynchronize(0);
    ec.chk("cudaInvXFormXZ");

    block_size = dim3(8, 8, 8);
    grid_size = dim3(nx,ny,nz);
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    grid_size.z /= 4;

    cout << grid_size.x << " " << grid_size.y << " " << grid_size.z << endl;
    cudaInvXFormZY<Int> << <grid_size, block_size >> >
        (
        raw_pointer_cast(q.data())
        );
    cudaStreamSynchronize(0);
    ec.chk("cudaInvXFormZY");

}
template<class Int, class UInt, class Scalar, uint bsize>
void gpuTest
(
device_vector<Scalar> &data,
device_vector<Int> &q,
device_vector<UInt> &buffer
)
{
    host_vector<int> h_emax;
    host_vector<Scalar> h_p;
    host_vector<Int> h_q;
    host_vector<UInt> h_buf;
    host_vector<Bit<bsize> > h_bits;

    host_vector<Scalar> h_data;
    h_data = data;

    dim3 emax_size(nx / 4, ny / 4, nz / 4);

    dim3 block_size(8, 8, 8);
    dim3 grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

    device_vector<int> emax(emax_size.x * emax_size.y * emax_size.z);

    ErrorCheck ec;

    cudaEvent_t start_encode, stop_encode;
    float millisecs;

    cudaEventCreate(&start_encode);
    cudaEventCreate(&stop_encode);
    cudaEventRecord(start_encode, 0);


    ec.chk("pre-cudaMaxExp");
    cudaMaxExp << <grid_size, block_size >> >
        (
        raw_pointer_cast(emax.data()),
        raw_pointer_cast(data.data())
        );
    ec.chk("cudaMaxExp");

    ec.chk("pre-cudaFixedPoint");
    cudaFixedPoint << <grid_size, block_size >> >
        (
        raw_pointer_cast(emax.data()),
        raw_pointer_cast(data.data()),
        raw_pointer_cast(q.data())
        );
    ec.chk("cudaFixedPoint");

    gpuDecorrelate<Int>(q);
    block_size = dim3(8, 8, 8);
    grid_size = dim3(nx, ny, nz);
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    cudaint2uint << < grid_size, block_size >> >
        (
        raw_pointer_cast(q.data()),
        raw_pointer_cast(buffer.data())
        );
    ec.chk("cudaint2uint");

    //    q.clear();
    //    q.shrink_to_fit();

    device_vector<Bit<bsize> > stream(emax_size.x * emax_size.y * emax_size.z);


    block_size = dim3(8, 8, 8);
    grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    cudaencode<UInt, bsize> << < emax_size.x*emax_size.y*emax_size.z / 16, 16, 16 * (sizeof(Bit<bsize>) + sizeof(int)) >> >
        (
        raw_pointer_cast(buffer.data()),
        raw_pointer_cast(stream.data()),
        raw_pointer_cast(emax.data()),
        minbits, maxbits, maxprec, minexp, group_count, size
        );

    cudaEventRecord(stop_encode, 0);
    cudaEventSynchronize(stop_encode);
    cudaEventElapsedTime(&millisecs, start_encode, stop_encode);
    ec.chk("cudaencode");

    cout << "encode GPU in time (in ms): " << millisecs << endl;

    cudaEvent_t start_decode, stop_decode;

    block_size = dim3(8, 8, 8);
    grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    cudaRewind<bsize> << < grid_size, block_size >> >
        (
        raw_pointer_cast(stream.data())
        );
    ec.chk("cudaRewind");


    cudaEventCreate(&start_decode);
    cudaEventCreate(&stop_decode);
    cudaEventRecord(start_decode, 0);

    cudaMemset(thrust::raw_pointer_cast(buffer.data()),0, sizeof(UInt) * q.size());
    block_size = dim3(8,8,8);
    grid_size =  emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    cudaDecode<UInt, bsize><<< emax_size.x*emax_size.y*emax_size.z/16,16, 16*(sizeof(Bit<bsize>) + sizeof(int))>>>
        (
             raw_pointer_cast(buffer.data()),
             raw_pointer_cast(stream.data()),
             raw_pointer_cast(emax.data()),
             minbits, maxbits, maxprec, minexp, group_count, size
        );
    cudaStreamSynchronize(0);
    ec.chk("cudaDecode");

    cudaMemset(thrust::raw_pointer_cast(q.data()), 0, sizeof(Int)*q.size());

    block_size = dim3(8,8,8);
    grid_size = dim3(nx,ny,nz);
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    cudaInvOrder<<<grid_size, block_size>>>
        (
            raw_pointer_cast(buffer.data()),
            raw_pointer_cast(q.data())
        );
    cudaStreamSynchronize(0);
    ec.chk("cudaInvOrder");


//    block_size = dim3(8,8,8);
//    grid_size = emax_size;
//    grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

//    cudaInvXForm<Int><<<grid_size, block_size>>>
//        (
//            raw_pointer_cast(q.data())
//        );
//    cudaStreamSynchronize(0);
//    ec.chk("cudaInvXForm");
    gpuInvXform(q);

    cudaMemset(thrust::raw_pointer_cast(data.data()), 0, sizeof(Scalar)*data.size());
    block_size = dim3(8,8,8);
    grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

    cudaInvCast<Int, Scalar><<<grid_size, block_size>>>
            (
                raw_pointer_cast(emax.data()),
                raw_pointer_cast(data.data()),
                raw_pointer_cast(q.data())
                );
    ec.chk("cudaInvCast");

    cudaEventRecord(stop_decode, 0);
    cudaEventSynchronize(stop_decode);
    cudaEventElapsedTime(&millisecs, start_decode, stop_decode);
    ec.chk("cudaencode");

    cout << "decode GPU in time (in ms): " << millisecs << endl;

    host_vector<Scalar> data_out = data;

    for (int i = 0; i < nx*ny*nz; i++){
        if (h_data[i] != data_out[i]){
            cout << i << " " << h_data[i] << " " << data_out[i] << endl;
            exit(-1);
        }

        //assert(h_data[i] == data_out[i]);
    }
}

template<class Int, class UInt, class Scalar, uint bsize>
void cpuTest
(
host_vector<Scalar> &p

)
{
    uint mx = nx / 4;
    uint my = ny / 4;
    uint mz = nz / 4;
    host_vector<Bit<bsize> > stream(mx*my*mz);
    double start_time = omp_get_wtime();

#pragma omp parallel for
    for (int z = 0; z<nz; z += 4){
        for (int y = 0; y<ny; y += 4){
            for (int x = 0; x<nx; x += 4){
                int idx = z*nx*ny + y*nx + x;
                Int q2[64];
                UInt buf[64];

                int emax2 = max_exp<Scalar>(raw_pointer_cast(p.data()), x,y,z, 1, nx, nx*ny);
                fixed_point(q2, raw_pointer_cast(p.data()), emax2, x,y,z, 1, nx, nx*ny);
                fwd_xform<Int>(q2);
                reorder<Int, UInt>(q2, buf);
                //encode_ints<UInt>(stream[z/4 * mx*my + y/4 *mx + x/4], buf, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);
                encode_ints_par<UInt>(stream[z / 4 * mx*my + y / 4 * mx + x / 4], buf, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);
            }
        }
    }

    std::vector<Scalar> data_out(nx*ny*nz);

#pragma omp parallel for
    for (int z = 0; z < nz; z += 4){
        for (int y = 0; y < ny; y += 4){
            for (int x = 0; x < nx; x += 4){
                int idx = z*nx*ny + y*nx + x;

                stream[z / 4 * mx*my + y / 4 * mx + x / 4].rewind();
                int emax2 = max_exp<Scalar>(raw_pointer_cast(p.data()), x,y,z, 1, nx, nx*ny);

                UInt dec[64];
                decode_ints<UInt, bsize>(stream[z / 4 * mx*my + y / 4 * mx + x / 4], dec, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

                Int iblock[64];
                inv_order(dec, iblock, perm, 64);
                inv_xform(iblock);
                //inv_cast(iblock, &data_out[z*nx*ny+y*nx+x], emax2, 0, 0, 0, 1, 4, 16);

                Scalar fblock[64];
                inv_cast(iblock, fblock, emax2, 0, 0, 0, 1, 4, 16);
                for (int i = 0; i < 4; i++){
                    for (int j = 0; j < 4; j++){
                        for (int k = 0; k < 4; k++){
                            data_out[idx + i*nx*ny + j*nx + k] = fblock[i * 16 + j * 4 + k];

                        }
                    }
                }

            }
        }
    }


    double elapsed_time = omp_get_wtime() - start_time;
    cout << "CPU elapsed time (in secs): " << elapsed_time << endl;

    for (int i = 0; i < nx*ny*nz; i++){
        if (data_out[i] != p[i]){
            cout << i << " " << data_out[i] << " " << p[i] << endl;
            exit(-1);
        }
        //assert(data_out[i] == p[i]);
    }
}
int main()
{

    device_vector<double> d_vec_in(nx*ny*nz);
    device_vector<long long> d_vec_out(nx*ny*nz);
    device_vector<unsigned long long> d_vec_buffer(nx*ny*nz);
    host_vector<double> h_vec_in;

    thrust::counting_iterator<uint> index_sequence_begin(0);
    thrust::transform(
                    index_sequence_begin,
                    index_sequence_begin + nx*ny*nz,
                    d_vec_in.begin(),
                    RandGen());

    h_vec_in = d_vec_in;
//    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    setupConst<double>(perm);
    cout << "Begin gpuTestAll" << endl;
    gpuTest<long long, unsigned long long, double, 64>(d_vec_in, d_vec_out, d_vec_buffer);
   cout << "Finish gpuTestAll" << endl;
    cout << "Begin cpuTestAll" << endl;
    //testCPU<long long, unsigned long long, double, 64>(h_vec_in);
    cpuTest<long long, unsigned long long, double, 64>(h_vec_in);
    cout << "End cpuTestAll" << endl;
}
