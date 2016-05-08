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
using namespace cuZFP;

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const size_t nx = 128;
const size_t ny = 128;
const size_t nz = 128;

uint minbits = 1024;
uint maxbits = 1024;
uint MAXPREC = 64;
int MINEXP = -1074;
const double rate = 64;
size_t  blksize = 0;
unsigned long long group_count = 0x46acca631ull;
uint size = 64;
int EBITS = 11;                     /* number of exponent bits */


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
void setupConst(const unsigned char *perm,
	uint maxprec_,
	int minexp_,
	int ebits_)
{
	ErrorCheck ec;
	ec.chk("setupConst start");
	cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char) * 64, 0); ec.chk("setupConst: c_perm");

	const uint sizeof_scalar = sizeof(Scalar);
	cudaMemcpyToSymbol(c_sizeof_scalar, &sizeof_scalar, sizeof(uint)); ec.chk("setupConst: c_sizeof_scalar");

	cudaMemcpyToSymbol(c_maxprec, &maxprec_, sizeof(uint)); ec.chk("setupConst: c_maxprec");
	cudaMemcpyToSymbol(c_minexp, &minexp_, sizeof(int)); ec.chk("setupConst: c_minexp");
	cudaMemcpyToSymbol(c_ebits, &ebits_, sizeof(int)); ec.chk("setupConst: c_ebits");

	ec.chk("setupConst finished");



}

template<class Int, class UInt, class Scalar, uint bsize>
void cpuEFPDI2UTransform
(
const dim3 &emax_size,
const dim3 &blockDim,
const dim3 &gridDim,
const Scalar *data,
UInt *p,
Bit<bsize> *stream

)
{
	uint3 blockIdx;

	for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++){
		for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++){
			for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++){
				uint3 threadIdx;
				//extern __shared__ long long sh_q[];
				long long *sh_q = new long long[64*64];
				for (threadIdx.z = 0; threadIdx.z < blockDim.z; threadIdx.z++){
					for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++){
						for (threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x++){
							int mx = threadIdx.x + blockDim.x*blockIdx.x;
							int my = threadIdx.y + blockDim.y*blockIdx.y;
							int mz = threadIdx.z + blockDim.z*blockIdx.z;
							int eidx = mz*gridDim.x*blockDim.x*gridDim.y*blockDim.y + my*gridDim.x*blockDim.x + mx;

							mx *= 4; my *= 4; mz *= 4;
							//int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
							int emax = max_exp_block(data, mx, my, mz, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);

							stream[eidx].emax = emax;
							//	uint sz = gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4;
							//	uint sy = gridDim.x*blockDim.x * 4;
							//	uint sx = 1;
							fixed_point_block(sh_q + (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64, data, emax, mx, my, mz, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);
							fwd_xform(sh_q + (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64);


							//fwd_order
							for (int i = 0; i < 64; i++){
								uint idx = eidx * 64 + i;
								p[idx] = int2uint<Int, UInt>(sh_q[(threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64 + perm[i]]);
							}
						}
					}
				}
				delete[]sh_q;


			}
		}
	}
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


template<class UInt, uint bsize>
void cpuEncodeUInt
(
const unsigned long long count,
uint size,
const UInt* data,
const unsigned char *g_cnt,
Bit<bsize> *stream
)
{

	//extern __shared__ unsigned char smem[];
	//__shared__ unsigned char *sh_g, *sh_sbits;
	//__shared__ Bitter *sh_bitters;

	//sh_g = &smem[0];
	//sh_sbits = &smem[64];
	//sh_bitters = (Bitter*)&smem[64 + 64];

	//uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;

	//uint bidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*blockDim.x*blockDim.y*blockDim.z;

	uint3 blockIdx, gridDim, blockDim;
	gridDim.x = nx / 4;
	gridDim.y = ny / 4;
	gridDim.z = nz / 4;
	
	blockDim.x = blockDim.y = blockDim.z = 4;

	for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++){
		for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++){
			for (blockIdx.x = 0; blockIdx.x <gridDim.x; blockIdx.x++){
				uint bidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*blockDim.x*blockDim.y*blockDim.z;

				unsigned long long x[64];
				Bitter bitter[64];
				unsigned char sbit[64];
				for (int i = 0; i < 64; i++){
					bitter[i] = make_bitter(0, 0);
					sbit[i] = 0;
				}
				uint s_emax_bits[1];
				s_emax_bits[0] = 1;
				//maxprec, minexp, EBITS
				//	uint k = threadIdx.x + blockDim.x * blockIdx.x;
				int emax = stream[bidx / 64].emax;
				int maxprec = precision(emax, MAXPREC, MINEXP);
				int ebits = EBITS + 1;
				const uint kmin = intprec > maxprec ? intprec - maxprec : 0;

				uint e = maxprec ? emax + ebias : 0;
				//printf("%d %d %d %d\n", emax, maxprec, ebits, e);
				if (e){
					//write_bitters(bitter[0], make_bitter(2 * e + 1, 0), ebits, sbit[0]);
					stream[bidx / 64].begin[0] = 2 * e + 1;
					s_emax_bits[0] = ebits;
				}
//				const uint kmin = intprec > MAXPREC ? intprec - MAXPREC : 0;

				unsigned long long y[64];
				for (int tid = 0; tid < 64; tid++){
					/* extract bit plane k to x[k] */
					y[tid] = 0;
					for (uint i = 0; i < size; i++)
						y[tid] += ((data[bidx + i] >> tid) & (unsigned long long)1) << i;
					x[tid] = y[tid];
				}

				char sh_g[64], sh_sbits[64];
				Bitter sh_bitters[64];

				/* count number of positive group tests g[k] among 3*d in d dimensions */
				for (int tid = 0; tid < 64; tid++){
					sh_g[tid] = 0;
					for (unsigned long long c = count; y[tid]; y[tid] >>= c & 0xfu, c >>= 4)
						sh_g[tid]++;
				}


				unsigned char cur = sh_g[intprec - 1];

				for (int i = intprec - 1; i-- > kmin;) {
					if (cur < sh_g[i])
						cur = sh_g[i];
					else if (cur > sh_g[i])
						sh_g[i] = cur;
				}

				for (int tid = 0; tid < 64; tid++){
					unsigned char g = sh_g[tid];
					unsigned char h = sh_g[min(tid + 1, intprec - 1)];


					encodeBitplane(count, x[tid], g, h, g_cnt, bitter[tid], sbit[tid]);
					sh_bitters[63 - tid] = bitter[tid];
					sh_sbits[63 - tid] = sbit[tid];
				}


				uint tot_sbits = s_emax_bits[0];
				uint offset = 0;
				for (int i = 0; i < intprec; i++){
					if (sh_sbits[i] <= 64){
						write_outx(sh_bitters, stream[bidx / 64].begin, tot_sbits, offset, i, sh_sbits[i]);
					}
					else{
						write_outx(sh_bitters, stream[bidx / 64].begin, tot_sbits, offset, i, 64);
						write_outy(sh_bitters, stream[bidx / 64].begin, tot_sbits, offset, i, sh_sbits[i] - 64);
					}
				}
			}
		}
	}
}


template<class Int, class UInt, class Scalar, uint bsize>
void cpuEncode
(
dim3 gridDim, 
dim3 blockDim,
const unsigned long long count,
uint size,
const Scalar* data,
const unsigned char *g_cnt,
Bit<bsize> *stream
)
{

	//extern __shared__ unsigned char smem[];
	//__shared__ unsigned char *sh_g, *sh_sbits;
	//__shared__ Bitter *sh_bitters;

	//sh_g = &smem[0];
	//sh_sbits = &smem[64];
	//sh_bitters = (Bitter*)&smem[64 + 64];

	//uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;

	//uint bidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*blockDim.x*blockDim.y*blockDim.z;

	dim3 blockIdx;
	
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++){
		for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++){
			for (blockIdx.x = 0; blockIdx.x <gridDim.x; blockIdx.x++){


				Int sh_q[64];
				UInt sh_p[64];
				uint mx = blockIdx.x, my = blockIdx.y, mz = blockIdx.z;
				mx *= 4; my *= 4; mz *= 4;
				int emax = max_exp_block(data, mx, my, mz, 1, blockDim.x * gridDim.x, gridDim.x * gridDim.y * blockDim.x * blockDim.y);

				//	uint sz = gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4;
				//	uint sy = gridDim.x*blockDim.x * 4;
				//	uint sx = 1;
				fixed_point_block(sh_q, data, emax, mx, my, mz, 1, blockDim.x * gridDim.x, gridDim.x  * gridDim.y * blockDim.x * blockDim.y);
				fwd_xform(sh_q);


				//fwd_order
				for (int i = 0; i < 64; i++){
					sh_p[i] = int2uint<Int, UInt>(sh_q[perm[i]]);
				}


				uint bidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);

				unsigned long long x[64];
				Bitter bitter[64];
				unsigned char sbit[64];
				for (int i = 0; i < 64; i++){
					bitter[i] = make_bitter(0, 0);
					sbit[i] = 0;
				}
				uint s_emax_bits[1];
				s_emax_bits[0] = 1;
				//maxprec, minexp, EBITS
				//	uint k = threadIdx.x + blockDim.x * blockIdx.x;
				int maxprec = precision(emax, MAXPREC, MINEXP);
				int ebits = EBITS + 1;
				const uint kmin = intprec > maxprec ? intprec - maxprec : 0;

				uint e = maxprec ? emax + ebias : 0;
				//printf("%d %d %d %d\n", emax, maxprec, ebits, e);
				if (e){
					//write_bitters(bitter[0], make_bitter(2 * e + 1, 0), ebits, sbit[0]);
					stream[bidx].begin[0] = 2 * e + 1;
					s_emax_bits[0] = ebits;
				}
				//				const uint kmin = intprec > MAXPREC ? intprec - MAXPREC : 0;

				unsigned long long y[64];
				for (int tid = 0; tid < 64; tid++){
					/* extract bit plane k to x[k] */
					y[tid] = 0;
					for (uint i = 0; i < size; i++)
						y[tid] += ((sh_p[i] >> tid) & (unsigned long long)1) << i;
					x[tid] = y[tid];
				}

				char sh_g[64], sh_sbits[64];
				Bitter sh_bitters[64];

				/* count number of positive group tests g[k] among 3*d in d dimensions */
				for (int tid = 0; tid < 64; tid++){
					sh_g[tid] = 0;
					for (unsigned long long c = count; y[tid]; y[tid] >>= c & 0xfu, c >>= 4)
						sh_g[tid]++;
				}


				unsigned char cur = sh_g[intprec - 1];

				for (int i = intprec - 1; i-- > kmin;) {
					if (cur < sh_g[i])
						cur = sh_g[i];
					else if (cur > sh_g[i])
						sh_g[i] = cur;
				}

				for (int tid = 0; tid < 64; tid++){
					unsigned char g = sh_g[tid];
					unsigned char h = sh_g[min(tid + 1, intprec - 1)];


					encodeBitplane(count, x[tid], g, h, g_cnt, bitter[tid], sbit[tid]);
					sh_bitters[63 - tid] = bitter[tid];
					sh_sbits[63 - tid] = sbit[tid];
				}


				uint tot_sbits = s_emax_bits[0];
				uint offset = 0;
				for (int i = 0; i < intprec; i++){
					if (sh_sbits[i] <= 64){
						write_outx(sh_bitters, stream[bidx].begin, tot_sbits, offset, i, sh_sbits[i]);
					}
					else{
						write_outx(sh_bitters, stream[bidx].begin, tot_sbits, offset, i, 64);
						write_outy(sh_bitters, stream[bidx].begin, tot_sbits, offset, i, sh_sbits[i] - 64);
					}
				}
			}
		}
	}
}
template<class Int, class UInt, uint bsize, uint num_sidx>
void cpuDecodeInvOrder
(
size_t *sidx,
Bit<bsize> *stream,

Int *data,

const unsigned long long orig_count

)
{
	//uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	//uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	//uint bdim = blockDim.x*blockDim.y*blockDim.z;
	//uint bidx = idx*bdim;

	uint3 blockIdx, gridDim, blockDim;
	gridDim.x = nx / 4;
	gridDim.y = ny / 4;
	gridDim.z = nz / 4;

	blockDim.x = blockDim.y = blockDim.z = 4;

	for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++){
		for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++){
			for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++){
				uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
				uint bdim = blockDim.x*blockDim.y*blockDim.z;
				uint bidx = idx*bdim;

				size_t s_sidx[64];// = (size_t*)&smem[0];
				//if (tid < num_sidx)
				for (int tid = 0; tid < num_sidx; tid++){

					s_sidx[tid] = sidx[tid];
				}

				uint s_idx_n[64];// = (uint*)&smem[s_sidx[0]];
				uint s_idx_g[64];// = (uint*)&smem[s_sidx[1]];
				unsigned long long s_bit_cnt[64];// = (unsigned long long*)&smem[s_sidx[2]];
				uint s_bit_rmn_bits[64];// = (uint*)&smem[s_sidx[3]];
				char s_bit_offset[64];// = (char*)&smem[s_sidx[4]];
				uint s_bit_bits[64];// = (uint*)&smem[s_sidx[5]];
				Word s_bit_buffer[64];// = (Word*)&smem[s_sidx[6]];
				UInt s_data[64];// = (UInt*)&smem[s_sidx[7]];
				uint s_kmin[1];


				stream[idx].read_bit();
				uint ebits = EBITS + 1;
				int emax = stream[idx].read_bits(ebits - 1) - ebias;
				int maxprec = precision(emax, MAXPREC, MINEXP);
				s_kmin[0] = intprec > maxprec ? intprec - maxprec : 0;

				
				for (int tid = 0; tid < 64; tid++){
					s_idx_g[tid] = 0;
					s_data[tid] = 0;
				}

				insert_bit<bsize>(
					stream[idx],
					s_idx_g,
					s_idx_n,
					s_bit_bits,
					s_bit_offset,
					s_bit_buffer,
					s_bit_cnt,
					s_bit_rmn_bits,
					maxbits - ebits, intprec, s_kmin[0], orig_count);

				for (int tid = 0; tid < 64; tid++){

					for (uint k = s_kmin[0]; k < intprec; k++){
						decodeBitstream<UInt, bsize>(
							stream[idx],
							s_idx_g[k],
							s_idx_n[k],
							s_bit_cnt[k],
							s_bit_rmn_bits[k],
							s_bit_bits[k],
							s_bit_offset[k],
							s_bit_buffer[k],
							s_data[tid],
							tid, k);
					}

					data[perm[tid] + bidx] = uint2int<Int, UInt>(s_data[tid]);
				}
			}
		}
	}
}

template<class Int, class UInt, class Scalar, uint bsize, uint num_sidx>
void cpuDecode
(
dim3 gridDim,
dim3 blockDim,
size_t *sidx,
Bit<bsize> *stream,

Scalar *out,
const unsigned long long orig_count

)
{
	//uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	//uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	//uint bdim = blockDim.x*blockDim.y*blockDim.z;
	//uint bidx = idx*bdim;

	dim3 blockIdx;

	for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++){
		for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++){
			for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++){
				uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
				uint bdim = blockDim.x*blockDim.y*blockDim.z;
				uint bidx = idx*bdim;

				size_t s_sidx[64];// = (size_t*)&smem[0];
				//if (tid < num_sidx)
				for (int tid = 0; tid < num_sidx; tid++){

					s_sidx[tid] = sidx[tid];
				}

				uint s_idx_n[64];// = (uint*)&smem[s_sidx[0]];
				uint s_idx_g[64];// = (uint*)&smem[s_sidx[1]];
				unsigned long long s_bit_cnt[64];// = (unsigned long long*)&smem[s_sidx[2]];
				uint s_bit_rmn_bits[64];// = (uint*)&smem[s_sidx[3]];
				char s_bit_offset[64];// = (char*)&smem[s_sidx[4]];
				uint s_bit_bits[64];// = (uint*)&smem[s_sidx[5]];
				Word s_bit_buffer[64];// = (Word*)&smem[s_sidx[6]];
				UInt s_data[64];// = (UInt*)&smem[s_sidx[7]];
				Int s_q[64];
				uint s_kmin[1];
				int s_emax[1];


				stream[idx].read_bit();
				uint ebits = EBITS + 1;
				s_emax[0] = stream[idx].read_bits(ebits - 1) - ebias;
				int maxprec = precision(s_emax[0], MAXPREC, MINEXP);
				s_kmin[0] = intprec > maxprec ? intprec - maxprec : 0;


				for (int tid = 0; tid < 64; tid++){
					s_idx_g[tid] = 0;
					s_data[tid] = 0;
				}

				insert_bit<bsize>(
					stream[idx],
					s_idx_g,
					s_idx_n,
					s_bit_bits,
					s_bit_offset,
					s_bit_buffer,
					s_bit_cnt,
					s_bit_rmn_bits,
					maxbits - ebits, intprec, s_kmin[0], orig_count);

				for (int tid = 0; tid < 64; tid++){

					for (uint k = s_kmin[0]; k < intprec; k++){
						decodeBitstream<UInt, bsize>(
							stream[idx],
							s_idx_g[k],
							s_idx_n[k],
							s_bit_cnt[k],
							s_bit_rmn_bits[k],
							s_bit_bits[k],
							s_bit_offset[k],
							s_bit_buffer[k],
							s_data[tid],
							tid, k);
					}

					s_q[perm[tid]] = uint2int<Int, UInt>(s_data[tid]);




				}

				uint mx = blockIdx.x, my = blockIdx.y, mz = blockIdx.z;
				mx *= 4; my *= 4; mz *= 4;

				inv_xform(s_q);
				inv_cast<Int, Scalar>(s_q, out, s_emax[0], mx, my, mz, 1, gridDim.x*blockDim.x, gridDim.x*blockDim.x * gridDim.y*blockDim.y);

			}
		}
	}
}
template<class Int, class UInt, class Scalar, uint bsize>
void gpuValidate
(
host_vector<Scalar> &h_p,
device_vector<Int> &q,
device_vector<Scalar> &data
)
{
	host_vector<Int> h_q;

	h_q = q;

	int i = 0;
	for (int z = 0; z < nz; z += 4){
		for (int y = 0; y < ny; y += 4){
			for (int x = 0; x < nx; x += 4){
				int idx = z*nx*ny + y*nx + x;
				host_vector<Int> q2(64);
				host_vector<UInt> buf(64);
				Bit<bsize> loc_stream;
				int emax2 = max_exp_block<Scalar>(raw_pointer_cast(h_p.data()), x, y, z, 1, nx, nx*ny);
				fixed_point_block(raw_pointer_cast(q2.data()), raw_pointer_cast(h_p.data()), emax2, x, y, z, 1, nx, nx*ny);
				fwd_xform(raw_pointer_cast(q2.data()));
				reorder<Int, UInt>(raw_pointer_cast(q2.data()), raw_pointer_cast(buf.data()));
				encode_ints<UInt>(loc_stream, raw_pointer_cast(buf.data()), minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

				loc_stream.rewind();
				UInt dec[64];

				decode_ints<UInt, bsize>(loc_stream, dec, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);


				Int iblock[64];
				inv_order(dec, iblock, perm, 64);
				inv_xform(iblock);

				for (int j = 0; j < 64; j++){
					assert(h_q[i * 64 + j] == iblock[j]);
				}

				Scalar fblock[64];
				inv_cast(iblock, fblock, emax2, 0, 0, 0, 1, 4, 16);

				int fidx = 0;
				for (int k = z; k < z + 4; k++){
					for (int j = y; j < y + 4; j++){
						for (int i = x; i < x + 4; i++, fidx++){
							if (h_p[k*nz*ny + j*ny + i] != fblock[fidx]){
								cout << "inv_cast failed: " << k << " " << j << " " << i << " " << fidx << " " << h_p[k*nz*ny + j*ny + i] << " " << fblock[fidx] << endl;
								exit(-1);
							}

						}
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
	host_vector<int> h_emax;
	host_vector<UInt> h_p;
	host_vector<Int> h_q;
	host_vector<UInt> h_buf(nx*ny*nz);
	host_vector<Bit<bsize> > h_bits;
	device_vector<unsigned char> d_g_cnt;

  device_vector<Scalar> data;
  data = h_data;


	dim3 emax_size(nx / 4, ny / 4, nz / 4);

	dim3 block_size(8, 8, 8);
	dim3 grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	//const uint kmin = intprec > maxprec ? intprec - maxprec : 0;

	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);



	device_vector<Bit<bsize> > stream(emax_size.x * emax_size.y * emax_size.z);
	host_vector<Bit<bsize> > cpu_stream;

	block_size = dim3(4, 4, 4);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;
	
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
#if 0
	cpu_stream = stream;
	cpuEncode<Int, UInt, Scalar, bsize>(
		grid_size,
		block_size,
		group_count, size,
		thrust::raw_pointer_cast(h_data.data()),
		thrust::raw_pointer_cast(g_cnt.data()),
		thrust::raw_pointer_cast(cpu_stream.data()));
	stream = cpu_stream;
#else
	cudaEncode<Int, UInt,Scalar, bsize> << <grid_size, block_size, (2 * sizeof(unsigned char) + sizeof(Bitter) + sizeof(UInt) + sizeof(Int)) * 64 + 4 >> >
		(
		group_count, size,
		thrust::raw_pointer_cast(data.data()),
		thrust::raw_pointer_cast(d_g_cnt.data()),
		thrust::raw_pointer_cast(stream.data())
		);
	 cudaStreamSynchronize(0);
	ec.chk("cudaEncode");
	cpu_stream = stream;
#endif
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs << endl;

  cudaMemset(thrust::raw_pointer_cast(data.data()), 0, sizeof(Scalar)*data.size());

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaRewind<bsize> << < grid_size, block_size >> >
		(
		raw_pointer_cast(stream.data())
		);
	ec.chk("cudaRewind");

#if 1
	block_size = dim3(4, 4, 4);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	size_t blcksize = block_size.x *block_size.y * block_size.z;
#else
	host_vector<size_t> cpu_sidx = d_sidx;
	host_vector<Int> cpu_q = q;
	cpuDecodeInvOrder < Int, UInt, bsize, 9 >
		(
		raw_pointer_cast(cpu_sidx.data()),
			raw_pointer_cast(cpu_stream.data()),
			raw_pointer_cast(cpu_q.data()),
			group_count);
	stream = cpu_stream;
	q = cpu_q;
#endif

#if 1
	size_t s_idx[12] = { sizeof(size_t) * 12, blcksize * sizeof(uint), blcksize * sizeof(uint), +blcksize * sizeof(unsigned long long), blcksize * sizeof(uint), blcksize * sizeof(char), blcksize * sizeof(uint), blcksize * sizeof(Word), blcksize * sizeof(UInt), blcksize * sizeof(Int), sizeof(uint), sizeof(int) };
	thrust::inclusive_scan(s_idx, s_idx + 11, s_idx);
	const size_t shmem_size = thrust::reduce(s_idx, s_idx + 11);
	device_vector<size_t> d_sidx(s_idx, s_idx + 11);

	cudaDecode<Int, UInt, Scalar, bsize, 11> << < grid_size, block_size, 64 * (4 + 4 + 8 + 4 + 1 + 4 + 8 + 8 + 8) + 4 + 4 >> >

		(
		raw_pointer_cast(d_sidx.data()),
		raw_pointer_cast(stream.data()),
		raw_pointer_cast(data.data()),
		maxbits,
		intprec,
		group_count);
	cudaStreamSynchronize(0);
#else
	cpuDecode < Int, UInt, Scalar, bsize, 9 >
		(grid_size, block_size,
		raw_pointer_cast(cpu_sidx.data()),
		raw_pointer_cast(cpu_stream.data()),
		raw_pointer_cast(h_data.data()),
		group_count);
	data = h_data;
#endif
  ec.chk("cudaDecodeInvOrder");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudadecode");
	cout << "decode parallel GPU in time: " << millisecs << endl;

	host_vector<Scalar> h_out = data;
	for (int i = 0; i < h_data.size(); i++){
		if (h_data[i] != h_out[i]){
			cout << i << " " << h_data[i] << " " << h_out[i] << endl;
			exit(-1);
		}
	}
	//gpuValidate<Int, UInt, Scalar, bsize>(h_data, q, data);

}

int main()
{

  host_vector<double> h_vec_in(nx*ny*nz);
  for (int z=0; z<nz; z++){
    for (int y=0; y<ny; y++){
      for (int x=0; x<nx; x++){
        if (x == 0)
          h_vec_in[z*nx*ny + y*nx + x] = 10;
        else if(x == nx - 1)
          h_vec_in[z*nx*ny + y*nx + x] = 0;
        else
          h_vec_in[z*nx*ny + y*nx + x] = 5;

      }
    }
  }
  device_vector<double> d_vec_in;
  d_vec_in = h_vec_in;

//	thrust::counting_iterator<uint> index_sequence_begin(0);
//	thrust::transform(
//		index_sequence_begin,
//		index_sequence_begin + nx*ny*nz,
//		d_vec_in.begin(),
//		RandGen());

	h_vec_in = d_vec_in;
	d_vec_in.clear();
	d_vec_in.shrink_to_fit();
	//    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	setupConst<double>(perm, MAXPREC, MINEXP, EBITS);
	cout << "Begin gpuTestBitStream" << endl;
	gpuTestBitStream<long long int, unsigned long long int, double, 64>(h_vec_in);
	cout << "Finish gpuTestBitStream" << endl;
	//    cout << "Begin cpuTestBitStream" << endl;
	//    cpuTestBitStream<long long, unsigned long long, double, 64>(h_vec_in);
	//    cout << "End cpuTestBitStream" << endl;

	//cout << "Begin gpuTestHarnessSingle" << endl;
	//gpuTestharnessSingle<long long, unsigned long long, double, 64>(h_vec_in, d_vec_out, d_vec_in, 0,0,0);
	//cout << "Begin gpuTestHarnessMulti" << endl;
	//gpuTestharnessMulti<long long, unsigned long long, double, 64>(d_vec_in);
}
