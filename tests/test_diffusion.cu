#include <iostream>
#include <iomanip>
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

#include "zfparray3.h"

using namespace thrust;
using namespace std;

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const size_t nx = 64;
const size_t ny = 64;
const size_t nz = 64;
const int nt = 0;


//BSIZE is the length of the array in class Bit
//It's tied to MAXBITS such that 
//MAXBITS = sizeof(Word) * BSIZE
//which is really
//MAXBITS = wsize * BSIZE
//e.g. if we match bits one-to-one, double -> unsigned long long
// then BSIZE = 64 and MAXPBITS = 4096
#define BSIZE  16
uint minbits = 1024;
uint MAXBITS = 1024;
uint MAXPREC = 64;
int MINEXP = -1074;
const double rate = 16;
size_t  blksize = 0;
unsigned long long group_count = 0x46acca631ull;
uint size = 64;
int EBITS = 11;                     /* number of exponent bits */
const int EBIAS = 1023;
const int intprec = 64;


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
	uint maxbits_,
	uint maxprec_,
	int minexp_,
	int ebits_,
	int ebias_
	)
{
	ErrorCheck ec;
	ec.chk("setupConst start");
	cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char) * 64, 0); ec.chk("setupConst: c_perm");

	cudaMemcpyToSymbol(c_maxbits, &MAXBITS, sizeof(uint)); ec.chk("setupConst: c_maxbits");
	const uint sizeof_scalar = sizeof(Scalar);
	cudaMemcpyToSymbol(c_sizeof_scalar, &sizeof_scalar, sizeof(uint)); ec.chk("setupConst: c_sizeof_scalar");

	cudaMemcpyToSymbol(c_maxprec, &maxprec_, sizeof(uint)); ec.chk("setupConst: c_maxprec");
	cudaMemcpyToSymbol(c_minexp, &minexp_, sizeof(int)); ec.chk("setupConst: c_minexp");
	cudaMemcpyToSymbol(c_ebits, &ebits_, sizeof(int)); ec.chk("setupConst: c_ebits");
	cudaMemcpyToSymbol(c_ebias, &ebias_, sizeof(int)); ec.chk("setupConst: c_ebias");

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

// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_zy(Int* p)
{
	for (uint z = 0; z < 4; z++)
		for (uint y = 4; y-- > 0;)
	       cuZFP::fwd_lift<Int, 1>(p + 4 * y + 16 * z);

}
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_xz(Int* p)
{
	for (uint x = 4; x-- > 0;)
	  for (uint z = 4; z-- > 0;)
			cuZFP::fwd_lift<Int, 4>(p + 16 * z + 1 * x);

}
// forward decorrelating transform
template<class Int>
__host__
static void
fwd_xform_yx(Int* p)
{
	for (uint y = 4; y-- > 0;)
	     for (uint x = 4; x-- > 0;)
				 cuZFP::fwd_lift<Int, 16>(p + 1 * x + 4 * y);

}

// forward decorrelating transform
template<class Int>
__host__
static void
fwd_xform(Int* p)
{
	fwd_xform_zy(p);
	fwd_xform_xz(p);
	fwd_xform_yx(p);
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
Word *block
)
{

	dim3 blockIdx;
	
	for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++){
		for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++){
			for (blockIdx.x = 0; blockIdx.x <gridDim.x; blockIdx.x++){

				Int sh_q[64];
				UInt sh_p[64];
				uint sh_m[64], sh_n[64];
				Bitter sh_bitters[64];
				unsigned char sh_sbits[64];

				uint mx = blockIdx.x, my = blockIdx.y, mz = blockIdx.z;
				mx *= 4; my *= 4; mz *= 4;
				int emax = cuZFP::max_exp_block(data, mx, my, mz, 1, blockDim.x * gridDim.x, gridDim.x * gridDim.y * blockDim.x * blockDim.y);

				//	uint sz = gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4;
				//	uint sy = gridDim.x*blockDim.x * 4;
				//	uint sx = 1;
				cuZFP::fixed_point_block<Int, Scalar, intprec>(sh_q, data, emax, mx, my, mz, 1, blockDim.x * gridDim.x, gridDim.x  * gridDim.y * blockDim.x * blockDim.y);
				fwd_xform(sh_q);


				//fwd_order
				for (int i = 0; i < 64; i++){
					sh_p[i] = cuZFP::int2uint<Int, UInt>(sh_q[perm[i]]);
				}


				uint bidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);

        //cuZFP::Bit<bsize> stream(block + bidx*bsize);

				unsigned long long x[64], y[64];
				Bitter bitter[64];
				for (int i = 0; i < 64; i++){
					bitter[i] = make_bitter(0, 0);
				}
				uint s_emax_bits[1];
				s_emax_bits[0] = 1;
				//maxprec, minexp, EBITS
				//	uint k = threadIdx.x + blockDim.x * blockIdx.x;
				int maxprec = cuZFP::precision(emax, MAXPREC, MINEXP);
				int ebits = EBITS + 1;
				const uint kmin = intprec > maxprec ? intprec - maxprec : 0;

				uint e = maxprec ? emax + EBIAS : 0;
				//printf("%d %d %d %d\n", emax, maxprec, ebits, e);
				if (e){
					//write_bitters(bitter[0], make_bitter(2 * e + 1, 0), ebits, sbit[0]);
          block[bidx*bsize] = 2 * e + 1;
					//stream[bidx].write_bits(2 * e + 1, ebits);
					s_emax_bits[0] = ebits;
				}
				//				const uint kmin = intprec > MAXPREC ? intprec - MAXPREC : 0;

				//unsigned long long x[64];

#pragma omp parallel for
				for (int tid = 0; tid<64; tid++){
					/* step 1: extract bit plane #k to x */
					x[tid] = 0;
					for (int i = 0; i < size; i++)
						x[tid] += (uint64)((sh_p[i] >> tid) & 1u) << i;
					y[tid] = x[tid];
				}

#pragma omp parallel for
				for (int tid = 0; tid < 64; tid++){
					sh_m[tid] = 0;
					sh_n[tid] = 0;
					sh_sbits[tid] = 0;
				}

#pragma omp parallel for
				for (int tid = 0; tid < 64; tid++){
					//get the index of the first 'one' in the bit plane
					for (int i = 0; i < 64; i++){
						if (!!(x[tid] >> i))
							sh_n[tid] = i + 1;
					}
				}
				for (int i = 0; i < 63; i++){
					sh_m[i] = sh_n[i + 1];
				}

				//make sure that m increases isotropically
				for (int i = intprec - 1; i-- > 0;){
					if (sh_m[i] < sh_m[i + 1])
						sh_m[i] = sh_m[i + 1];
				}				

				//compute the number of bits used per thread
				int bits[64];
#pragma omp parallel for
				for (int tid = 0; tid < 64; tid++) {
					bits[tid] = 128;
					int n = 0;
					/* step 2: encode first n bits of bit plane */
					bits[tid] -= sh_m[tid];
					x[tid] >>= sh_m[tid];
					x[tid] = (sh_m[tid] != 64) * x[tid];
					n = sh_m[tid];
					/* step 3: unary run-length encode remainder of bit plane */
					for (; n < size && bits[tid] && (bits[tid]--, !!x[tid]); x[tid] >>= 1, n++)
						for (; n < size - 1 && bits[tid] && (bits[tid]--, !(x[tid] & 1u)); x[tid] >>= 1, n++)
							;
				}

				//number of bits read per thread
//#pragma omp parallel for
				for (int tid = 0; tid < 64; tid++){
					bits[tid] = (128 - bits[tid]);
				}
#pragma omp parallel for
				for (int tid = 0; tid < 64; tid++){
					sh_n[tid] = min(sh_m[tid], bits[tid]);
				}

#pragma omp parallel for
				for (int tid = 0; tid < 64; tid++) {
					/* step 2: encode first n bits of bit plane */
					unsigned char sbits = 0;
					//y[tid] = stream[bidx].write_bits(y[tid], sh_m[tid]);
					y[tid] = write_bitters(bitter[tid], make_bitter(y[tid], 0), sh_m[tid], sbits);
					uint n = sh_n[tid];

					/* step 3: unary run-length encode remainder of bit plane */
					for (; n < size && bits[tid] && (bits[tid]-- && write_bitter(bitter[tid], !!y[tid], sbits)); y[tid] >>= 1, n++)
						for (; n < size - 1 && bits[tid] && (bits[tid]-- && !write_bitter(bitter[tid], y[tid] & 1u, sbits)); y[tid] >>= 1, n++)
							;

					sh_bitters[63 - tid] = bitter[tid];
					sh_sbits[63 - tid] = sbits;
				}

				uint rem_sbits = s_emax_bits[0];
				uint tot_sbits = s_emax_bits[0];
				uint offset = 0;
				for (int i = 0; i < intprec && tot_sbits < MAXBITS; i++){
					if (sh_sbits[i] <= 64){
            write_outx<bsize>(sh_bitters, block + bidx*bsize, rem_sbits, tot_sbits, offset, i, sh_sbits[i]);
					}
					else{
            write_outx<bsize>(sh_bitters, block + bidx*bsize, rem_sbits, tot_sbits, offset, i, 64);
            if (tot_sbits < MAXBITS)
              write_outy<bsize>(sh_bitters, block + bidx*bsize, rem_sbits, tot_sbits, offset, i, sh_sbits[i] - 64);
					}
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
Word *block,

Scalar *out,
const unsigned long long orig_count

)
{

	dim3 blockIdx;

	for (blockIdx.z = 0; blockIdx.z < gridDim.z; blockIdx.z++){
		for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++){
			for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++){
				uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
				uint bdim = blockDim.x*blockDim.y*blockDim.z;
				uint bidx = idx*bdim;

        cuZFP::Bit<bsize> stream;
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

        //stream[idx].rewind();

        stream.read_bit();
				uint ebits = EBITS + 1;
				s_emax[0] = stream.read_bits(ebits - 1) - EBIAS;
				int maxprec = cuZFP::precision(s_emax[0], MAXPREC, MINEXP);
				s_kmin[0] = intprec > maxprec ? intprec - maxprec : 0;

				for (int tid = 0; tid < size; tid++)
					s_data[tid] = 0;

				uint bits = MAXBITS - ebits;

				unsigned long long x[64];


        int *sh_idx = new int[bsize*64];
				int *sh_tmp_idx = new int[bsize * 64];


				for (int tid = 0; tid < 64; tid++){
					for (int i = 0; i < 16; i++){
						sh_idx[i * 64 + tid] = -1;
						sh_tmp_idx[i * 64 + tid] = -1;
					}
				}

				int sh_cnt[bsize];
				int beg_idx[bsize];
				for (int tid = 0; tid < 64; tid++){
					if (tid < bsize){
						beg_idx[tid] = 0;
						if (tid == 0)
							beg_idx[tid] = ebits;
						sh_cnt[tid] = 0;
						for (int i = beg_idx[tid]; i < 64; i++){
              if ((stream.begin[tid] >> i) & 1u){
								sh_tmp_idx[tid * 64 + sh_cnt[tid]++] = tid*64 + i;
							}
						}
					}
				}

				//fix blocks since they are off by ebits
				for (int i = 0; i < bsize; i++){
					for (int tid = 0; tid < 64; tid++){
						if (tid < sh_cnt[i]){
							sh_tmp_idx[i*64 + tid] -= ebits;
						}
					}
				}

				for (int tid = 0; tid < 64; tid++){
					if (tid < sh_cnt[0])
						sh_idx[tid] = sh_tmp_idx[tid];
				}

				for (int i = 1; i < bsize; i++){
					for (int tid = 0; tid < 64; tid++){
						if (tid == 0)
							sh_cnt[i] += sh_cnt[i - 1];
						if (tid < sh_cnt[i]){
							sh_idx[sh_cnt[i - 1] + tid] = sh_tmp_idx[i * 64 + tid];
						}
					}
				}


				/* decode one bit plane at a time from MSB to LSB */
        int cnt = 0;
				//uint new_n = 0;
				uint bits_cnt = ebits;
				for (uint tid = intprec, n = 0; bits && tid-- > s_kmin[0];) {
					/* decode first n bits of bit plane #k */
					uint m = MIN(n, bits);
					bits -= m;
					bits_cnt += m;
          x[tid] = stream.read_bits(m);
					/* unary run-length decode remainder of bit plane */
          for (; n < size && bits && (bits--, bits_cnt++, stream.read_bit()); x[tid] += (uint64)1 << n++){
						int num_bits = 0;
						uint chk = 0;

						//uint tmp_bits = stream[idx].bits;
						//Word tmp_buffer = stream[idx].buffer;
						//char tmp_offset = stream[idx].offset;
            //for (; n < size - 1 && bits && (bits--, !stream[idx].read_bit()); n++)
            //  ;
						//stream[idx].bits = tmp_bits;
						//stream[idx].buffer = tmp_buffer;
						//stream[idx].offset = tmp_offset;

            while (n < size - 1 && bits && (bits--, bits_cnt++, !stream.read_bit())){
							//the number of bits read in one go: 
							//this can be affected by running out of bits in the block (variable bits)
							// and how much is encoded per number (variable n)
							// and how many zeros there are since the last one bit.
							// Finally, the last bit isn't read because we'll check it to see 
							// where we are

							/* fast forward to the next one bit that hasn't been read yet*/
							while (sh_idx[cnt] < bits_cnt - ebits){
                cnt++;
              }
							cnt--;
							//compute the raw number of bits between the last one bit and the current one bit
							num_bits = sh_idx[cnt + 1] - sh_idx[cnt];

							//the one bit as two positions previous
							num_bits -= 2;

							num_bits = min(num_bits, (size - 1) - n - 1);

							bits_cnt += num_bits;
							if (num_bits > 0){
                stream.read_bits(num_bits);
								bits -= num_bits;
								n += num_bits;
							}

							n++;
						}
            //if (n != new_n || new_bits != bits){
            //   cout << n << " " << new_n << " " << bits << " " << new_bits << " " << blockIdx.x * gridDim.x << " " << blockIdx.y*gridDim.y << " " << blockIdx.z * gridDim.z << endl;
            //  exit(0);
            //}
          }
					/* deposit bit plane from x */
					for (int i = 0; x[tid]; i++, x[tid] >>= 1)
						s_data[i] += (UInt)(x[tid] & 1u) << tid;


				}

				for (int tid = 0; tid < 64; tid++){
					s_q[perm[tid]] = cuZFP::uint2int<Int, UInt>(s_data[tid]);

				}


				uint mx = blockIdx.x, my = blockIdx.y, mz = blockIdx.z;
				mx *= 4; my *= 4; mz *= 4;

				cuZFP::inv_xform(s_q);
				cuZFP::inv_cast<Int, Scalar>(s_q, out, s_emax[0], mx, my, mz, 1, gridDim.x*blockDim.x, gridDim.x*blockDim.x * gridDim.y*blockDim.y);

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
	host_vector<cuZFP::Bit<bsize> > h_bits;
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



  device_vector<Word > block(emax_size.x * emax_size.y * emax_size.z * bsize);
  cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, data, block, group_count, size);

	cudaStreamSynchronize(0);
	ec.chk("cudaEncode");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs << endl;

  thrust::host_vector<Word > cpu_block;
  cpu_block = block;
	UInt sum = 0;
  for (int i = 0; i < cpu_block.size(); i++){
    sum += cpu_block[i];
	}
	cout << "encode UInt sum: " << sum << endl;

  cudaMemset(thrust::raw_pointer_cast(data.data()), 0, sizeof(Scalar)*data.size());

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);


	cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, block, data, group_count);

  ec.chk("cudaDecode");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);

	cout << "decode parallel GPU in time: " << millisecs << endl;

	double tot_sum = 0, max_diff = 0, min_diff = 1e16;

	host_vector<Scalar> h_out = data;
	for (int i = 0; i < h_data.size(); i++){
		int k = 0, j = 0;
		frexp(h_data[i], &j);
		frexp(h_out[i], &k);

		//if (abs(j - k) > 1){
		//	cout << i << " " << j << " " << k << " " << h_data[i] << " " << h_out[i] << endl;
		//	//exit(-1);
		//}
		double diff = fabs(h_data[i] - h_out[i]);
		//if (diff > 1 )
		//	cout << i << " " << j << " " << k << " " << h_data[i] << " " << h_out[i] << endl;

		if (max_diff < diff)
			max_diff = diff;
		if (min_diff > diff)
			min_diff = diff;

		tot_sum += diff;
	}

	cout << "tot diff: " << tot_sum << " average diff: " << tot_sum / (float)h_data.size() << " max diff: " << max_diff << " min diff: " << min_diff << endl;
	cout << "sum: " << thrust::reduce(h_data.begin(), h_data.end()) << " " << thrust::reduce(h_out.begin(), h_out.end()) << endl;

	//gpuValidate<Int, UInt, Scalar, bsize>(h_data, q, data);

}
void discrete_solution
(
zfp::array3d &u,

int x0, int y0, int z0,
const double dx,
const double dy,
const double dz,
const double dt,
const double k,
const double tfinal
)
{
	// initialize u (constructor zero-initializes)
	//rate = u.rate();
	u(x0, y0, z0) = 1;

	// iterate until final time
	std::cerr.precision(6);
	double t;
	for (t = 0; t < tfinal; t += dt) {
		std::cerr << "t=" << std::fixed << t << std::endl;
		// compute du/dt
		zfp::array3d du(nx, ny, nz, rate);
		for (int z = 1; z < nz - 1; z++){
			for (int y = 1; y < ny - 1; y++) {
				for (int x = 1; x < nx - 1; x++) {
					double uxx = (u(x - 1, y, z) - 2 * u(x, y, z) + u(x + 1, y, z)) / (dx * dx);
					double uyy = (u(x, y - 1, z) - 2 * u(x, y, z) + u(x, y + 1, z)) / (dy * dy);
					double uzz = (u(x, y, z - 1) - 2 * u(x, y, z) + u(x, y, z + 1)) / (dz * dz);
					du(x, y, z) = dt * k * (uxx + uyy + uzz);
				}
			}
		}
		// take forward Euler step
		for (uint i = 0; i < u.size(); i++)
			u[i] += du[i];
	}
}
void rme
(
	const zfp::array3d &u,
	int x0,
	int y0,
	int z0,
	const double dx,
	const double dy,
	const double dz,
	const double k,
	const double pi,
	double t
)
{

	// compute root mean square error with respect to exact solution
	double e = 0;
	double sum = 0;
	for (int z = 1; z < nz - 1; z++){
		double pz = dz * (z - z0);
		for (int y = 1; y < ny - 1; y++) {
			double py = dy * (y - y0);
			for (int x = 1; x < nx - 1; x++) {
				double px = dx * (x - x0);
				double f = u(x, y, z);
				//http://nptel.ac.in/courses/105103026/34
				double g = dx * dy * dz * std::exp(-(px * px + py * py + pz * pz) / (4 * k * t)) / powf(4 * pi * k * t, 3.0/2.0);
				e += (f - g) * (f - g);
				sum += f;
			}
		}
	}

	e = std::sqrt(e / ((nx - 2) * (ny - 2)));
	std::cerr.unsetf(std::ios::fixed);
	std::cerr << "rate=" << rate << " sum=" << std::fixed << sum << " error=" << std::setprecision(6) << std::scientific << e << std::endl;

}
int main()
{
	host_vector<double> h_vec_in(nx*ny*nz);

	device_vector<double> d_vec_in(nx*ny*nz);
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
	setupConst<double>(perm, MAXBITS, MAXPREC, MINEXP, EBITS, EBIAS);
	cout << "Begin gpuTestBitStream" << endl;
  gpuTestBitStream<long long, unsigned long long, double, BSIZE>(h_vec_in);
	cout << "Finish gpuTestBitStream" << endl;



	// location of point heat source
	int x0 = (nx - 1) / 2;
	int y0 = (ny - 1) / 2;
	int z0 = (nz - 1) / 2;
	// constants used in the solution
	const double k = 0.04;
	const double dx = 2.0 / (std::max(nz,std::max(nx, ny)) - 1);
	const double dy = 2.0 / (std::max(nz, std::max(nx, ny)) - 1);
	const double dz = 2.0 / (std::max(nz, std::max(nx, ny)) - 1);
	const double dt = 0.5 * (dx * dx + dy * dy) / (8 * k);
	const double tfinal = nt ? nt * dt : 1;
	const double pi = 3.14159265358979323846;

	zfp::array3d u(nx, ny, nz, rate);

	discrete_solution(u, x0, y0, z0, dx,dy,dz,dt,k, tfinal);

	rme(u, x0, y0, z0, dx, dy, dz, k, pi, tfinal-dt);

}
