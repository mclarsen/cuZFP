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
#include "include/BitStream.cuh"
#include "include/ull128.h"
using namespace thrust;
using namespace std;

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const size_t nx = 512;
const size_t ny = 512;
const size_t nz = 512;

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

__device__ __host__
Bitter subull2(Bitter in1, Bitter in2)
{
	Bitter difference;
	difference.y = in1.y - in2.y;
	difference.x = in1.x - in2.x;
	// check for underflow of low 64 bits, subtract carry to high
	if (difference.y > in1.x)
		--difference.y;
	return difference;
}

unsigned long long
__device__ __host__
write_bitters(Bitter &bitters, Bitter value, uint n, unsigned char &sbits)
{
	if (n == bitsize(value.x)){
		bitters.x = value.x;
		bitters.y = value.y;
		sbits += n;
		return 0;
	}
	else{
		Bitter v = rshiftull2(value, n);
		Bitter ret = rshiftull2(value, n);
		v = lshiftull2(v, n);
		value = subull2(value, v);

		v = lshiftull2(value, sbits);
		bitters.x += v.x;
		bitters.y += v.y;

		sbits += n;
		return ret.x;
	}
}

__device__ __host__
void
write_bitter(Bitter &bitters, Bitter bit, unsigned char &sbits)
{
	Bitter val = lshiftull2(bit, sbits++);
	bitters.x += val.x;
	bitters.y += val.y;
}

__device__ __host__
void
write_out(unsigned long long *out, uint &tot_sbits, uint &offset, unsigned long long value, uint sbits)
{

	out[offset] += value << tot_sbits;
	tot_sbits += sbits;
	if (tot_sbits >= wsize) {
		tot_sbits -= wsize;
		offset++;
		if (tot_sbits > 0)
			out[offset] = value >> (sbits - tot_sbits);
	}
}


__shared__ Bitter sh_bitters[64];
__device__ __host__
void
write_outx(unsigned long long *out, uint &tot_sbits, uint &offset, unsigned long idx, uint sbits)
{

  out[offset] += sh_bitters[idx].x << tot_sbits;
  tot_sbits += sbits;
  if (tot_sbits >= wsize) {
    tot_sbits -= wsize;
    offset++;
    if (tot_sbits > 0)
      out[offset] = sh_bitters[idx].x >> (sbits - tot_sbits);
  }
}
__device__ __host__
void
write_outy(unsigned long long *out, uint &tot_sbits, uint &offset, unsigned long idx, uint sbits)
{

  out[offset] += sh_bitters[idx].y << tot_sbits;
  tot_sbits += sbits;
  if (tot_sbits >= wsize) {
    tot_sbits -= wsize;
    offset++;
    if (tot_sbits > 0)
      out[offset] = sh_bitters[idx].y >> (sbits - tot_sbits);
  }
}


inline
__device__ __host__
void
encodeBitplane
(
const uint kmin,
unsigned long long count,

unsigned long long x,
const unsigned char g,
unsigned char h,
const unsigned char *g_cnt,

//uint &h, uint &n_cnt, unsigned long long &cnt,
Bitter &bitters,
unsigned char &sbits

)
{
	unsigned long long cnt = count;
	cnt >>= h * 4;
	uint n_cnt = g_cnt[h];

	/* serial: output one bit plane at a time from MSB to LSB */
	bitters.x = 0;
	bitters.y = 0;

	sbits = 0;
	/* encode bit k for first n values */
	x = write_bitters(bitters, make_bitter(x, 0), n_cnt, sbits);
	while (h++ < g) {
		/* output a one bit for a positive group test */
		write_bitter(bitters, make_bitter(1, 0), sbits);
		/* add next group of m values to significant set */
		uint m = cnt & 0xfu;
		cnt >>= 4;
		n_cnt += m;
		/* encode next group of m values */
		x = write_bitters(bitters, make_bitter(x, 0), m, sbits);
	}
	/* if there are more groups, output a zero bit for a negative group test */
	if (cnt) {
		write_bitter(bitters, make_bitter(0, 0), sbits);
	}
}
__global__
void
cudaEncodeBitplane
(
const uint kmin,
unsigned long long count,

unsigned long long *x,
const unsigned char *g,
const unsigned char *g_cnt,

//uint *h, uint *n_cnt, unsigned long long *cnt,
Bitter *bitters,
unsigned char *sbits
)
{
	uint k = threadIdx.x + blockDim.x * blockIdx.x;
	extern __shared__ unsigned long long sh_x[];

	sh_x[threadIdx.x] = x[k];
	__syncthreads();
	encodeBitplane(kmin, count, x[k], g[k], g[blockDim.x*blockIdx.x + min(threadIdx.x + 1, intprec - 1)], g_cnt, bitters[blockDim.x *(blockIdx.x + 1) - threadIdx.x - 1], sbits[blockDim.x*(blockIdx.x + 1) - threadIdx.x - 1]);
}

template<class UInt, uint bsize>
__global__
void cudaEncode
(
uint kmin,
const unsigned long long count,
uint size,
const UInt* data,
const unsigned char *g_cnt,
Bit<bsize> *stream
)
{

	extern __shared__ unsigned char sh_g[], sh_sbits[];
	unsigned long long x;


  uint k = threadIdx.x + blockDim.x * blockIdx.x;

	/* extract bit plane k to x[k] */
	unsigned long long y = 0;
	for (uint i = 0; i < size; i++)
		y += ((data[blockDim.x*blockIdx.x+i] >> threadIdx.x) & (unsigned long long)1) << i;
	x = y;

	__syncthreads();
	/* count number of positive group tests g[k] among 3*d in d dimensions */
	sh_g[threadIdx.x] = 0;
	for (unsigned long long c = count; y; y >>= c & 0xfu, c >>= 4)
		sh_g[threadIdx.x]++;

	__syncthreads();
	if (threadIdx.x == 0){
		unsigned char cur = sh_g[intprec-1];

		for (int i = intprec - 1; i-- > kmin;) {
			if (cur < sh_g[i])
				cur = sh_g[i];
			else if (cur > sh_g[i])
				sh_g[i] = cur;
		}
	}

	__syncthreads();
//	g[k] = sh_g[threadIdx.x];

	unsigned char g = sh_g[threadIdx.x];
	unsigned char h = sh_g[min(threadIdx.x + 1, intprec - 1)];

	Bitter bitter = make_bitter(0, 0);
	unsigned char sbit = 0;
	encodeBitplane(kmin, count, x, g, h, g_cnt,bitter, sbit);
  sh_bitters[63-threadIdx.x] = bitter;
	sh_sbits[63 - threadIdx.x] = sbit;

	__syncthreads();
	uint idx = blockDim.x * blockIdx.x;
	if (threadIdx.x == 0){
		uint tot_sbits = 0;// sbits[0];
		uint offset = 0;
		for (int i = 0; i < intprec; i++){
			if (sh_sbits[i] <= 64){
        write_outx(stream[idx / 64].begin, tot_sbits, offset, i, sh_sbits[i]);
			}
			else{
        write_outx(stream[idx / 64].begin, tot_sbits, offset, i, 64);
        write_outy(stream[idx / 64].begin, tot_sbits, offset, i, sh_sbits[i] - 64);
			}
		}
	}
}

template<class UInt>
__global__
void cudaEncodeGroup
(
unsigned long long *x,
unsigned char *g,
const UInt* data,
unsigned long long count,
uint size
)
{
	uint k = threadIdx.x + blockDim.x * blockIdx.x;
	extract_bit(threadIdx.x, x[k], g[blockDim.x*blockIdx.x + 64 - threadIdx.x - 1], data + blockDim.x*blockIdx.x, count, size);
}

template<class UInt>
__global__
void cudaGroupScan
(
unsigned char *g,
uint intprec,
uint kmin
)
{
	extern __shared__ unsigned char sh_g[], sh_gout[];
	uint idx = (blockDim.x * blockIdx.x + threadIdx.x);
	//thrust::inclusive_scan(thrust::device, g + k, g + k + 64, g + k, thrust::maximum<uint>());

	sh_g[threadIdx.x] = g[idx];
	__syncthreads();
	if (threadIdx.x == 0){
		unsigned char cur = sh_g[0];
		sh_gout[63] = cur;

		for (int i = 0; i < 64; i++) {
			if (cur < sh_g[i])
				cur = sh_g[i];
			else if (cur > sh_g[i])
				sh_g[i] = cur;
		}
	}

	__syncthreads();
	g[idx] = sh_g[63-threadIdx.x];
}

template<uint bsize>
__global__
void cudaCompact
(
const uint intprec,
Bit<bsize> *stream,
const unsigned char *sbits,
const Bitter *bitters
)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	uint tot_sbits = 0;// sbits[0];
	uint offset = 0;


	for (int k = 0; k < intprec; k++){
		if (sbits[idx * 64 + k] <= 64){
			write_out(stream[idx].begin, tot_sbits, offset, bitters[idx * 64 + k].x, sbits[idx * 64 + k]);
		}
		else{
			write_out(stream[idx].begin, tot_sbits, offset, bitters[idx * 64 + k].x, 64);
			write_out(stream[idx].begin, tot_sbits, offset, bitters[idx * 64 + k].y, sbits[idx * 64 + k] - 64);
		}
	}
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


template<class Int, class UInt, class Scalar, uint bsize>
void gpuTestBitStream
(
host_vector<Scalar> &h_data
)
{
	device_vector<Int> q(nx*ny*nz);
	device_vector<UInt> buffer(nx*ny*nz);
	device_vector<Scalar> data = h_data;
  device_vector<unsigned char> d_g_cnt;

	dim3 emax_size(nx / 4, ny / 4, nz / 4);

	dim3 block_size(8, 8, 8);
	dim3 grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	device_vector<int> emax(emax_size.x * emax_size.y * emax_size.z);

	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


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


	data.clear();
	data.shrink_to_fit();
	cudaDecorrelate<Int> << <grid_size, block_size >> >
		(
		raw_pointer_cast(q.data())
		);
	ec.chk("cudaDecorrelate");

	block_size = dim3(8, 8, 8);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaint2uint << < grid_size, block_size >> >
		(
		raw_pointer_cast(q.data()),
		raw_pointer_cast(buffer.data())
		);
	ec.chk("cudaint2uint");

	//q.clear();
	//q.shrink_to_fit();

	device_vector<Bit<bsize> > stream(emax_size.x * emax_size.y * emax_size.z);


	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;


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


  cudaEncode<UInt, bsize> << <nx*ny*nz / 64, 64, 2*(sizeof(unsigned char) + sizeof(unsigned long long)) * 64 >> >
		(
		kmin, group_count, size,
		thrust::raw_pointer_cast(buffer.data()),
		thrust::raw_pointer_cast(d_g_cnt.data()),
    thrust::raw_pointer_cast(stream.data())
		);
	cudaStreamSynchronize(0);
	ec.chk("cudaEncode");

  //device_vector<unsigned long long> d_x(nx*ny*nz);
	//device_vector<unsigned char> d_g(nx*ny*nz)
	//cudaEncodeGroup<UInt> << <nx*ny*nz / 64, 64 >> >(thrust::raw_pointer_cast(d_x.data()), thrust::raw_pointer_cast(d_g.data()), thrust::raw_pointer_cast(buffer.data()), group_count, size);
	//ec.chk("cudaEncodeGroup");

	//cudaGroupScan<UInt> << <nx*ny*nz / (64), 64, sizeof(unsigned char) * 2 * 64 >> >(thrust::raw_pointer_cast(d_g.data()), intprec, kmin);
	//ec.chk("cudaGroupScan");

	//host_vector<unsigned char> h_g = d_g;
	//
	//ec.chk("pre encodeBitplane");
	//cudaEncodeBitplane << <  nx*ny*nz / 64, 64, (sizeof(uint) + sizeof(unsigned long long)) * 64 >> >
	//	(
	//	kmin, group_count,
	//	thrust::raw_pointer_cast(d_x.data()),
	//	thrust::raw_pointer_cast(d_g.data()),
	//	thrust::raw_pointer_cast(d_g_cnt.data()),
	//	thrust::raw_pointer_cast(d_bitters.data()),
	//	thrust::raw_pointer_cast(d_sbits.data())
	//	);
	//cudaStreamSynchronize(0);
	//ec.chk("cudaEncodeBitplane");

	//cudaCompact<bsize> << <nx*nz*nz / (64 * 64), 64 >> >(intprec, thrust::raw_pointer_cast(stream.data()), thrust::raw_pointer_cast(d_sbits.data()), thrust::raw_pointer_cast(d_bitters.data()));
	//cudaStreamSynchronize(0);
	//ec.chk("cudaCompact");
	//host_vector<Bitter> h_bitters = d_bitters;
	//host_vector<unsigned char> h_sbits = d_sbits;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs << endl;

	gpuValidate<Int, UInt, Scalar, bsize>(h_data, buffer, stream);

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
	//    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	setupConst(perm);
	cout << "Begin gpuTestBitStream" << endl;
	gpuTestBitStream<long long, unsigned long long, double, 64>(h_vec_in);
	cout << "Finish gpuTestBitStream" << endl;
}
