﻿#include <iostream>
#include <helper_math.h>
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
#include "include/BitStream.cuh"
#include "include/ull128.h"
using namespace thrust;
using namespace std;

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const size_t nx = 128;
const size_t ny = 128;
const size_t nz = 128;

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


void setupConst(const unsigned char *perm)
{
	ErrorCheck ec;
	ec.chk("setupConst start");
	cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char) * 64, 0); ec.chk("setupConst: lic_dim");
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



template<class Int, class UInt, class Scalar, uint bsize>
void gpuValidate
(
host_vector<Scalar> &h_p,
device_vector<UInt> &buffer,
device_vector<Bit<bsize> > &stream
)
{
	host_vector<UInt> h_buf;
	host_vector<Bit<bsize> > h_bits;

	h_buf = buffer;
	h_bits = stream;

	int i = 0;
	for (int z = 0; z < nz; z += 4){
		for (int y = 0; y < ny; y += 4){
			for (int x = 0; x < nx; x += 4){
				int idx = z*nx*ny + y*nx + x;
				host_vector<Int> q2(64);
				host_vector<UInt> buf(64);
				Bit<bsize> loc_stream;
				int emax2 = max_exp<Scalar>(raw_pointer_cast(h_p.data()), x, y, z, 1, nx, nx*ny);
				//fixed_point(raw_pointer_cast(q2.data()),raw_pointer_cast(h_p.data()), emax2, idx, 1,nx,nx*ny);
				fixed_point(raw_pointer_cast(q2.data()), raw_pointer_cast(h_p.data()), emax2, x, y, z, 1, nx, nx*ny);

				fwd_xform(raw_pointer_cast(q2.data()));
				reorder<Int, UInt>(raw_pointer_cast(q2.data()), raw_pointer_cast(buf.data()));
				encode_ints<UInt>(loc_stream, raw_pointer_cast(buf.data()), minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

				for (int j = 0; j < 64; j++){
					if (h_bits[i].begin[j] != loc_stream.begin[j]){
						cout << x << " " << y << " " << z << " " << j << " " << h_bits[i].begin[j] << " " << loc_stream.begin[j] << endl;
						exit(-1);
					}
				}

				i++;

			}
		}
	}
}

template<class Int, class Scalar>
__global__
void cudaInvEmaxXformCast
(
Scalar *data,
Int *iblock
)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	int eidx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;

	x *= 4; y *= 4; z *= 4;

	inv_xform(iblock + eidx * 64);
	int emax = max_exp(data, x, y, z, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);
	inv_cast<Int, Scalar>(iblock + eidx * 64, data, emax, x, y, z, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);

}

template<class Int, class UInt, uint bsize>
__global__
void cudaDecodeInvOrder
(
Bit<bsize> *stream,

Int *data,

const uint maxbits,
const uint intprec,
const uint kmin,
const unsigned long long orig_count

)
{
	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bdim = blockDim.x*blockDim.y*blockDim.z;
	uint bidx = idx*bdim;

	extern __shared__ unsigned char smem[];
	uint *s_idx_n = (uint*)&smem[0];
	uint *s_idx_g = (uint*)&smem[64 * sizeof(uint)];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[64 * (4 + 4)];
	uint *s_bit_rmn_bits = (uint*)&smem[64 * (4 + 4 + 8)];
	char *s_bit_offset = (char*)&smem[64 * (4 + 4 + 8 + 4)];
	uint *s_bit_bits = (uint*)&smem[64 * (4 + 4 + 8 + 4 + 1)];
	Word *s_bit_buffer = (Word*)&smem[64 * (4 + 4 + 8 + 4 + 1 + 4)];
	UInt *s_data = (UInt*)&smem[64 * (4 + 4 + 8 + 4 + 1 + 4 + 8)];
	s_idx_g[tid] = 0;
	__syncthreads();

	if (tid == 0){
		insert_bit<bsize>(
			stream[idx],
			s_idx_g,
			s_idx_n,
			s_bit_bits,
			s_bit_offset,
			s_bit_buffer,
			s_bit_cnt,
			s_bit_rmn_bits,
			maxbits, intprec, kmin, orig_count);
	}	__syncthreads();

	for (uint k = kmin; k < intprec; k++){
		decodeBitstream<UInt, bsize>(
			stream[idx],
			s_idx_g[k],
			s_idx_n[k],
			s_bit_cnt[k],
			s_bit_rmn_bits[k],
			s_bit_bits[k],
			s_bit_offset[k],
			s_bit_buffer[k],
			s_data,
			maxbits, intprec, kmin, tid, k);
	}

	data[c_perm[tid] + bidx] = uint2int<Int, UInt>(s_data[tid]);

}

template<class Int, class UInt, class Scalar, uint bsize>
void gpuTestBitStream
(
host_vector<Scalar> &h_data
)
{
	device_vector<UInt> buffer(nx*ny*nz);
	device_vector<Scalar> data = h_data;
  device_vector<unsigned char> d_g_cnt;

	dim3 emax_size(nx / 4, ny / 4, nz / 4);

	dim3 block_size(4, 4, 4);
	dim3 grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	ec.chk("pre-cudaEFPDI2UTransform");
	cudaEFPDI2UTransform <Int, UInt, Scalar> << < grid_size, block_size, sizeof(Int)*4*4*4* 4*4*4 >> >
		(
		raw_pointer_cast(data.data()),
		raw_pointer_cast(buffer.data())
		);
	ec.chk("post-cudaEFPDI2UTransform");

	device_vector<Bit<bsize> > stream(emax_size.x * emax_size.y * emax_size.z);




	const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
	unsigned long long count = group_count;
	host_vector<unsigned char> g_cnt(10);
	uint sum = 0;
	g_cnt[0] = 0;
	for (int i = 1; i < 10; i++){
		sum += count & 0xf;
		g_cnt[i] = sum;
		count >>= 4;
	}
	d_g_cnt = g_cnt;

	block_size = dim3(4, 4, 4);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

  cudaEncodeUInt<UInt, bsize> << <grid_size, block_size, 2*(sizeof(unsigned char) + sizeof(unsigned long long)) * 64 >> >
		(
		kmin, group_count, size,
		thrust::raw_pointer_cast(buffer.data()),
		thrust::raw_pointer_cast(d_g_cnt.data()),
    thrust::raw_pointer_cast(stream.data())
		);
	cudaStreamSynchronize(0);
	ec.chk("cudaEncode");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs << endl;

	buffer.clear();
	buffer.shrink_to_fit();
#define DEBUG
	device_vector<Int> q(nx*ny*nz);
#ifdef DEBUG
	cudaMemset(raw_pointer_cast(buffer.data()), 0, sizeof(UInt) * buffer.size());
	block_size = dim3(4, 4, 4);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaDecodePar<UInt, bsize> << < grid_size, block_size, 64 * 4 + 64 * 4 + 64 * 8 + 64 * 4 + 64 + 64 * 4 + sizeof(Word) * 64 >> >
		(
		raw_pointer_cast(stream.data()),
		raw_pointer_cast(buffer.data()),
		maxbits,
		intprec,
		kmin,
		group_count);
	cudaStreamSynchronize(0);
	ec.chk("cudaDecodePar");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudadecode");

	cout << "decode parallel GPU in time: " << millisecs << endl;
	block_size = dim3(8, 8, 8);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaInvOrder << <grid_size, block_size >> >
		(
		raw_pointer_cast(buffer.data()),
		raw_pointer_cast(q.data())
		);
	cudaStreamSynchronize(0);
	ec.chk("cudaInvOrder");
#else
	cudaMemset(raw_pointer_cast(q.data()), 0, sizeof(UInt) * buffer.size());
	block_size = dim3(4, 4, 4);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaDecodeInvOrder<Int, UInt, bsize> << <grid_size, block_size, 64 * 4 + 64 * 4 + 64 * 8 + 64 * 4 + 64 + 64 * 4 + sizeof(Word) * 64 + 64 * 8 >> >(
		raw_pointer_cast(stream.data()),
		raw_pointer_cast(q.data()),
		maxbits,
		intprec,
		kmin,
		group_count);
#endif

#ifdef DEBUG
	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;
	cudaInvXForm<Int> << <grid_size, block_size >> >
		(
		raw_pointer_cast(q.data())
		);
	cudaStreamSynchronize(0);
	ec.chk("cudaInvXForm");

	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	cudaInvCast<Int, Scalar> << <grid_size, block_size >> >
		(
		raw_pointer_cast(emax.data()),
		raw_pointer_cast(data.data()),
		raw_pointer_cast(q.data())
		);
	ec.chk("cudaInvCast");
#else	
	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	cudaInvEmaxXformCast<Int, Scalar> << <grid_size, block_size >> >(
		raw_pointer_cast(data.data()),
		raw_pointer_cast(q.data())
		);
	cudaStreamSynchronize(0);
	ec.chk("cudainvemaxxformcast");
#endif

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "decode GPU in time: " << millisecs << endl;

	for (int i = 0; i < data.size(); i++){
		if (h_data[i] != data[i]){
			cout << i << " " << h_data[i] << " " << data[i] << endl;
			exit(-1);
		}

	}
	//gpuValidate<Int, UInt, Scalar, bsize>(h_data, q, data);
	//gpuValidate<Int, UInt, Scalar, bsize>(h_data, buffer, stream);

}

int main()
{

	device_vector<double> d_vec_in(nx*ny*nz);
	host_vector<double> h_vec_in;

	thrust::counting_iterator<uint> index_sequence_begin(0);
	thrust::transform(
		index_sequence_begin,
		index_sequence_begin + nx*ny*nz,
		d_vec_in.begin(),
		RandGen());

	h_vec_in = d_vec_in;
	d_vec_in.clear();
	d_vec_in.shrink_to_fit();
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	setupConst(perm);
	cout << "Begin gpuTestBitStream" << endl;
	gpuTestBitStream<long long, unsigned long long, double, 64>(h_vec_in);
	cout << "Finish gpuTestBitStream" << endl;
}