#ifndef ENCODE_CUH
#define ENCODE_CUH

//#include <helper_math.h>
#include "shared.h"
#include "ull128.h"
#include "BitStream.cuh"
#include "WriteBitter.cuh"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#include <cuZFP.h>
#include <type_info.cuh>

#define LDEXP(x, e) ldexp(x, e)
#define FREXP(x, e) frexp(x, e)
#define FABS(x) fabs(x)

const int ebias = 1023;

namespace cuZFP{

// map two's complement signed integer to negabinary unsigned integer
inline __device__ __host__
unsigned long long int int2uint(const long long int x)
{
    return (x + (unsigned long long int)0xaaaaaaaaaaaaaaaaull) ^ 
                (unsigned long long int)0xaaaaaaaaaaaaaaaaull;
}

inline __device__ __host__
unsigned int int2uint(const int x)
{
    return (x + (unsigned int)0xaaaaaaaau) ^ 
                (unsigned int)0xaaaaaaaau;
}

template<class Scalar>
__host__ __device__
static int
exponent(Scalar x)
{
  if (x > 0) {
    int e;
    FREXP(x, &e);
    // clamp exponent in case x is denormalized
    return MAX(e, 1 - ebias);
  }
  return -ebias;
}


// lifting transform of 4-vector
template <class Int, uint s>
__device__ __host__
static void
fwd_lift(Int* p)
{
  Int x = *p; p += s;
  Int y = *p; p += s;
  Int z = *p; p += s;
  Int w = *p; p += s;

  // default, non-orthogonal transform (preferred due to speed and quality)
  //        ( 4  4  4  4) (x)
  // 1/16 * ( 5  1 -1 -5) (y)
  //        (-4  4  4 -4) (z)
  //        (-2  6 -6  2) (w)
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_zy(Int* p)
{
	fwd_lift<Int,1>(p + 4 * threadIdx.x + 16 * threadIdx.z);
}
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_xz(Int* p)
{
	fwd_lift<Int, 4>(p + 16 * threadIdx.z + 1 * threadIdx.x);
}
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_yx(Int* p)
{
	fwd_lift<Int, 16>(p + 1 * threadIdx.x + 4 * threadIdx.z);
}

// forward decorrelating transform
template<class Int>
__device__ 
static void
fwd_xform(Int* p)
{
  fwd_xform_zy(p);
	__syncthreads();
	fwd_xform_xz(p);
	__syncthreads();
	fwd_xform_yx(p);
}

template<typename Int, typename UInt, typename Scalar, int intprec>
__device__
void 
encode (Scalar *sh_data,
	      const uint bsize, 
        unsigned char *smem,
        uint blk_idx,
        Word *blocks)
{
  // number of bits in the incoming type
  const uint size = sizeof(Scalar) * 8; 
  const uint vals_per_block = 64;
  //shared mem that depends on scalar size
	__shared__ Scalar *sh_reduce;
	__shared__ Int *sh_q;
	__shared__ UInt *sh_p;

  // shared mem that always has the same size
	__shared__ int *sh_emax;
	__shared__ uint *sh_m, *sh_n;
	__shared__ unsigned char *sh_sbits;
	__shared__ Bitter *sh_bitters;
	__shared__ uint *s_emax_bits;

  //
  // These memory locations do not overlap
  // so we will re-use the same buffer to
  // conserve precious shared mem space
  //
	sh_reduce = &sh_data[0];
	sh_q = (Int*)&sh_data[0];
	sh_p = (UInt*)&sh_data[0];

	sh_sbits = &smem[0];
	sh_bitters = (Bitter*)&sh_sbits[vals_per_block];
	sh_m = (uint*)&sh_bitters[vals_per_block];
	sh_n = (uint*)&sh_m[vals_per_block];
	s_emax_bits = (uint*)&sh_n[vals_per_block];
	sh_emax = (int*)&s_emax_bits[1];

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;

	Bitter bitter = make_bitter(0, 0);
	unsigned char sbit = 0;
	//uint kmin = 0;
	if (tid < bsize)
		blocks[blk_idx + tid] = 0; 

  // cache the value this thread is resposible for
  Scalar thread_val = sh_data[tid];
	__syncthreads();
	//max_exp
	if (tid < 32)
		sh_reduce[tid] = max(fabs(sh_data[tid]), fabs(sh_data[tid + 32]));
	if (tid < 16)
		sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + 16]);
	if (tid < 8)
		sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + 8]);
	if (tid < 4)
		sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + 4]);
	if (tid < 2)
		sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + 2]);
	if (tid == 0){
		sh_reduce[0] = max(sh_reduce[tid], sh_reduce[tid + 1]);
		sh_emax[0] = exponent(sh_reduce[0]);
	}

  //if(tid == 0) printf(" sh_emax %d\n", sh_emax[0]);
  //if(tid == 0) printf(" max %f\n", sh_reduce[0]);
	__syncthreads();

	//fixed_point
	Scalar w = LDEXP(1.0, get_precision<Scalar>() - 2 - sh_emax[0]);
  // w = actu
  // sh_q  = signed integer representation of the floating point value
  // block tranform
  sh_q[tid] = (Int)(thread_val * w);
  // NO MORE sh_data

  // Decorrelation
	fwd_xform(sh_q);
	__syncthreads();
	//fwd_order
	sh_p[tid] = int2uint(sh_q[c_perm[tid]]);
  // sh_p negabinary rep
	if (tid == 0)
  {
    //   
		s_emax_bits[0] = 1;

		int maxprec = precision(sh_emax[0], c_maxprec, c_minexp);
		//kmin = intprec > maxprec ? intprec - maxprec : 0;

		uint e = maxprec ? sh_emax[0] + ebias : 0;
    //printf(" e %u\n", e);
		if (e)
    {
			//write_bitters(bitter[0], make_bitter(2 * e + 1, 0), ebits, sbit[0]);
			blocks[blk_idx] = 2 * e + 1; // the bit count?? for this block
			s_emax_bits[0] = c_ebits + 1;// this c_ebit = ebias
		}
	}
	__syncthreads();

	/* extract bit plane k to x[k] */
  // size  is probaly bit per value
	unsigned long long y = 0;
	for (uint i = 0; i < size; i++)
  {
		y += ((sh_p[i] >> tid) & (unsigned long long)1) << i;
  }
  // NO MORE sh_q
	unsigned long long x = y;

	__syncthreads();
	sh_m[tid] = 0; // is  
	sh_n[tid] = 0;

	//temporarily use sh_n as a buffer
	//uint *sh_test = sh_n;
  // these are setting up indices to things that have value
  // find the first 1 (in terms of most significant 
  // bit
	for (int i = 0; i < 64; i++)
  {
		if (!!(x >> i)) // !! is this bit zero
    {
			sh_n[tid] = i + 1;
    }
	}

	if (tid < 63)
  {
		sh_m[tid] = sh_n[tid + 1];
	}

	__syncthreads();
  // this is basically a scan
	if (tid == 0)
  {
		for (int i = intprec - 1; i-- > 0;)
    {
			if (sh_m[i] < sh_m[i + 1])
      {
				sh_m[i] = sh_m[i + 1];
      }
		}
	}
	__syncthreads();
  /// maybe don't use write bitter with float 32
	int bits = 128; // worst possible encoding outcome (128 is probably size of 'word')
	int n = 0;
	/* step 2: encode first n bits of bit plane */
	bits -= sh_m[tid];
	x >>= sh_m[tid];
	x = (sh_m[tid] != 64) * x;
	n = sh_m[tid];
	/* step 3: unary run-length encode remainder of bit plane */
	for (; n < size && bits && (bits--, !!x); x >>= 1, n++)
  {
		for (; n < size - 1 && bits && (bits--, !(x & 1u)); x >>= 1, n++);
  }
	__syncthreads();

	bits = (128 - bits);
	sh_n[tid] = min(sh_m[tid], bits);

	/* step 2: encode first n bits of bit plane */
	//y[tid] = stream[bidx].write_bits(y[tid], sh_m[tid]);
	y = write_bitters(bitter, make_bitter(y, 0), sh_m[tid], sbit);
	n = sh_n[tid];

	/* step 3: unary run-length encode remainder of bit plane */
	for (; n < size && bits && (bits-- && write_bitter(bitter, !!y, sbit)); y >>= 1, n++)
  {
		for (; n < size - 1 && bits && (bits-- && !write_bitter(bitter, y & 1u, sbit)); y >>= 1, n++);
  }
	__syncthreads();

  // First use of both bitters and sbits
	sh_bitters[63 - tid] = bitter;
	sh_sbits[63 - tid] = sbit;
	__syncthreads();

	if (tid == 0)
  {
		uint tot_sbits = s_emax_bits[0];// sbits[0];
		uint rem_sbits = s_emax_bits[0];// sbits[0];
		uint offset = 0;
    const uint maxbits = bsize * vals_per_block; 
		for (int i = 0; i < intprec && tot_sbits < maxbits; i++)
    {
			if (sh_sbits[i] <= 64)
      {
				write_outx(sh_bitters, blocks + blk_idx, rem_sbits, tot_sbits, offset, i, sh_sbits[i], bsize);
			}
			else
      {
        // I think  the 64 here is just capping out the bits it writes?
				write_outx(sh_bitters, blocks + blk_idx, rem_sbits, tot_sbits, offset, i, 64, bsize);
        if (tot_sbits < maxbits)
        {
          write_outy(sh_bitters, blocks + blk_idx, rem_sbits, tot_sbits, offset, i, sh_sbits[i] - 64, bsize);
        }
			}
		}
	}

}

template<class Int, class UInt, class Scalar, int intprec>
__global__
void
__launch_bounds__(64,5)
cudaEncode(const uint  bsize,
           const Scalar* data,
           Word *blocks,
           const int3 dims)
{
  //	int mx = threadIdx.x + blockDim.x*blockIdx.x;
  //	int my = threadIdx.y + blockDim.y*blockIdx.y;
  //	int mz = threadIdx.z + blockDim.z*blockIdx.z;
  //	int eidx = mz*gridDim.x*blockDim.x*gridDim.y*blockDim.y + my*gridDim.x*blockDim.x + mx;

  extern __shared__ unsigned char smem[];
	__shared__ Scalar *sh_data;
	unsigned char *new_smem;

	sh_data = (Scalar*)&smem[0];
	new_smem = (unsigned char*)&sh_data[64];

  uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
  uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);

  //
  //  The number of threads launched can be larger than total size of
  //  the array in cases where it cannot be devided into perfect block
  //  sizes. To account for this, we will clamp the values in each block
  //  to the bounds of the data set. 
  //

  const uint x_coord = min(threadIdx.x + blockIdx.x * 4, dims.x - 1);
  const uint y_coord = min(threadIdx.y + blockIdx.y * 4, dims.y - 1);
  const uint z_coord = min(threadIdx.z + blockIdx.z * 4, dims.z - 1);
      
	uint id = z_coord * dims.x * dims.y
          + y_coord * dims.x
          + x_coord;

	sh_data[tid] = data[id];

	__syncthreads();

	encode<Int, UInt, Scalar, intprec>(sh_data,
                                     bsize, 
                                     new_smem,
                                     idx * bsize,
                                     blocks);

  __syncthreads();

}


void allocate_device_mem3d(const int3 encoded_dims, 
                           const int bsize, 
                           thrust::device_vector<Word> &stream)
{
  const size_t vals_per_block = 64;
  const size_t size = encoded_dims.x * encoded_dims.y * encoded_dims.z; 
  size_t total_blocks = size / vals_per_block; 
  const size_t bits_per_block = vals_per_block * bsize;
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  const size_t alloc_size = total_bits / bits_per_word;
  stream.resize(alloc_size);
}

//
// Launch the encode kernel
//
template<class Int, class UInt, class Scalar, int intprec>
void encode (int3 dims, 
             const Scalar *d_data,
             thrust::device_vector<Word> &stream,
             const int bsize)
{
  dim3 block_size, grid_size;
  block_size = dim3(4, 4, 4);
  grid_size = dim3(dims.x, dims.y, dims.z);

  grid_size.x /= block_size.x; 
  grid_size.y /= block_size.y;  
  grid_size.z /= block_size.z;

  // Check to see if we need to increase the block sizes
  // in the case where dim[x] is not a multiple of 4

  int3 encoded_dims = dims;

  if(dims.x % 4 != 0) 
  {
    grid_size.x++;
    encoded_dims.x = grid_size.x * 4;
  }
  if(dims.y % 4 != 0) 
  {
    grid_size.y++;
    encoded_dims.y = grid_size.y * 4;
  }
  if(dims.z % 4 != 0)
  {
    grid_size.z++;
    encoded_dims.z = grid_size.z * 4;
  }

  allocate_device_mem3d(encoded_dims, bsize, stream);

  std::size_t shared_mem_size = sizeof(Scalar) * 64 +  sizeof(Bitter) * 64 + sizeof(unsigned char) * 64
                                + sizeof(unsigned int) * 128 + 2 * sizeof(int);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
	cudaEncode<Int, UInt, Scalar, intprec> << <grid_size, block_size, shared_mem_size>> >
    (bsize,
     d_data,
     thrust::raw_pointer_cast(stream.data()),
     dims);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  float rate = (float(dims.x * dims.y * dims.z) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Encode rate: %.2f (MB / sec)\n", rate);
}

//
// Just pass the raw pointer to the "real" encode
//
template<class Int, class UInt, class Scalar, int intprec>
void encode (int3 dims, 
             thrust::device_vector<Scalar> &d_data,
             thrust::device_vector<Word > &stream,
             const int bsize,
             const uint size)
{
  encode<Int, UInt, Scalar, intprec>(dims,
                                     thrust::raw_pointer_cast(d_data.data()),
                                     stream,
                                     bsize);
}

//
// Encode a host vector and output a encoded device vector
//
template<class Int, class UInt, class Scalar, int intprec>
void encode(int3 dims,
            const thrust::host_vector<Scalar> &h_data,
            thrust::device_vector<Word> &stream,
            const int bsize,
            const uint size)
{
  thrust::device_vector<Scalar> d_data = h_data;
  encode<Int, UInt, Scalar, intprec>(dims, d_data, stream, bsize, size);
}

//
//  Encode a host vector and output and encoded host vector
//
template<class Int, class UInt, class Scalar, int intprec>
void encode(int3 dims,
            const thrust::host_vector<Scalar> &h_data,
            thrust::host_vector<Word> &stream,
            const int bsize,
            const uint size)
{
  thrust::device_vector<Word > d_stream = stream;
  encode<Int, UInt, Scalar, intprec>(dims, h_data, d_stream, bsize, size);
  stream = d_stream;
}

}

#endif
