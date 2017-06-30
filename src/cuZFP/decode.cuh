#ifndef DECODE_CUH
#define DECODE_CUH

//#include <helper_math.h>
//dealing with doubles
#include "BitStream.cuh"
#include <thrust/device_vector.h>
#include <type_info.cuh>
#define NBMASK 0xaaaaaaaaaaaaaaaaull
#define LDEXP(x, e) ldexp(x, e)

namespace cuZFP {

#ifdef __CUDA_ARCH__
template<class Int, class Scalar>
__device__
Scalar
dequantize(Int x, int e)
{
	return LDEXP((double)x, e - (CHAR_BIT * scalar_sizeof<Scalar>() - 2));
}
#else
template<class Int, class Scalar, uint sizeof_scalar>
__host__
Scalar
dequantize(Int x, int e)
{
	return LDEXP((double)x, e - (CHAR_BIT * sizeof_scalar - 2));
}
#endif

/* inverse lifting transform of 4-vector */
template<class Int, uint s>
__host__ __device__
static void
inv_lift(Int* p)
{
	Int x, y, z, w;
	x = *p; p += s;
	y = *p; p += s;
	z = *p; p += s;
	w = *p; p += s;

	/*
	** non-orthogonal transform
	**       ( 4  6 -4 -1) (x)
	** 1/4 * ( 4  2  4  5) (y)
	**       ( 4 -2  4 -5) (z)
	**       ( 4 -6 -4  1) (w)
	*/
	y += w >> 1; w -= y >> 1;
	y += w; w <<= 1; w -= y;
	z += x; x <<= 1; x -= z;
	y += z; z <<= 1; z -= y;
	w += x; x <<= 1; x -= w;

	p -= s; *p = w;
	p -= s; *p = z;
	p -= s; *p = y;
	p -= s; *p = x;
}

/* transform along z */
template<class Int>
 __device__
static void
inv_xform_yx(Int* p)
{
	inv_lift<Int, 16>(p + 1 * threadIdx.x + 4 * threadIdx.z);
	//uint x, y;
	//for (y = 0; y < 4; y++)
	//	for (x = 0; x < 4; x++)
	//		inv_lift(p + 1 * x + 4 * y, 16);

}

/* transform along y */
template<class Int>
 __device__
static void
inv_xform_xz(Int* p)
{
	inv_lift<Int, 4>(p + 16 * threadIdx.z + 1 * threadIdx.x);
	//uint x, z;
	//for (x = 0; x < 4; x++)
	//	for (z = 0; z < 4; z++)
	//		inv_lift(p + 16 * z + 1 * x, 4);

}

/* transform along x */
template<class Int>
 __device__
static void
inv_xform_zy(Int* p)
{
	inv_lift<Int, 1>(p + 4 * threadIdx.x + 16 * threadIdx.z);
	//uint y, z;
	//for (z = 0; z < 4; z++)
	//	for (y = 0; y < 4; y++)
	//		inv_lift(p + 4 * y + 16 * z, 1);

}

/* inverse decorrelating 3D transform */
template<class Int>
 __device__
static void
inv_xform(Int* p)
{

	inv_xform_yx(p);
	__syncthreads();
	inv_xform_xz(p);
	__syncthreads();
	inv_xform_zy(p);
	__syncthreads();
}

/* map two's complement signed integer to negabinary unsigned integer */
inline __host__ __device__
long long int uint2int(unsigned long long int x)
{
	return (x ^0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
}

inline __host__ __device__
int uint2int(unsigned int x)
{
	return (x ^0xaaaaaaaau) - 0xaaaaaaaau;
}


__host__ __device__
int
read_bit(char &offset, uint &bits, Word &buffer, const Word *begin)
{
  uint bit;
  if (!bits) {
    buffer = begin[offset++];
    bits = wsize;
  }
  bits--;
  bit = (uint)buffer & 1u;
  buffer >>= 1;
  return bit;
}
/* read 0 <= n <= 64 bits */
__host__ __device__
unsigned long long
read_bits(uint n, char &offset, uint &bits, Word &buffer, const Word *begin)
{
#if 0
  /* read bits in LSB to MSB order */
  uint64 value = 0;
  for (uint i = 0; i < n; i++)
    value += (uint64)stream_read_bit(stream) << i;
  return value;
#elif 1
  uint BITSIZE = sizeof(unsigned long long) * CHAR_BIT;
  unsigned long long value;
  /* because shifts by 64 are not possible, treat n = 64 specially */
	if (n == BITSIZE) {
    if (!bits)
      value = begin[offset++];//*ptr++;
    else {
      value = buffer;
      buffer = begin[offset++];//*ptr++;
      value += buffer << bits;
      buffer >>= n - bits;
    }
  }
  else {
    value = buffer;
    if (bits < n) {
      /* not enough bits buffered; fetch wsize more */
      buffer = begin[offset++];//*ptr++;
      value += buffer << bits;
      buffer >>= n - bits;
      bits += wsize;
    }
    else
      buffer >>= n;
    value -= buffer << n;
    bits -= n;
  }
  return value;
#endif
}


template<typename Scalar>
__device__ 
Scalar  decode(const Word *blocks,
               unsigned char *smem,
               const uint bsize)
{
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;
  const int intprec = get_precision<Scalar>();
	__shared__ uint *s_kmin;
	__shared__ unsigned long long *s_bit_cnt;
	__shared__ Int *s_iblock;
	__shared__ int *s_emax;
	__shared__ int *s_cont;

	s_bit_cnt = (unsigned long long*)&smem[0];
	s_iblock = (Int*)&s_bit_cnt[0];
	s_kmin = (uint*)&s_iblock[64];

	s_emax = (int*)&s_kmin[1];
	s_cont = (int *)&s_emax[1];


	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	Scalar result = 0;

	if (tid == 0)
  {
		//Bit<bsize> stream(blocks + idx * bsize);
		uint sbits = 0;
		Word buffer = 0;
		char offset = 0;

		s_cont[0] = read_bit(offset, sbits, buffer, blocks);
	}

  __syncthreads();

	if (s_cont[0])
  {
		if (tid == 0)
    {
			uint sbits = 0;
			Word buffer = 0;
			char offset = 0;
			//do it again, it won't hurt anything
			read_bit(offset, sbits, buffer, blocks);

			uint ebits = get_ebits<Scalar>() + 1;
			s_emax[0] = read_bits(ebits - 1, offset, sbits, buffer, blocks) - get_ebias<Scalar>();
      //printf("Decode shemax %d\n", s_emax[0]);
			int maxprec = precision(s_emax[0], get_precision<Scalar>(), get_min_exp<Scalar>());
      //printf("max prec %d\n", maxprec);
			s_kmin[0] = intprec > maxprec ? intprec - maxprec : 0;
      const uint vals_per_block = 64;
      const uint maxbits = bsize * vals_per_block;
			uint bits = maxbits - ebits;
			for (uint k = intprec, n = 0; k-- > 0;)
      {
				//					idx_n[k] = n;
				//					bit_rmn_bits[k] = bits;
				uint m = MIN(n, bits);
				bits -= m;
				s_bit_cnt[k] = read_bits(m, offset, sbits, buffer, blocks);
				for (; n < 64 && bits && (bits--, read_bit(offset, sbits, buffer, blocks)); s_bit_cnt[k] += (unsigned long long)1 << n++)
					for (; n < 64 - 1 && bits && (bits--, !read_bit(offset, sbits, buffer, blocks)); n++)
						;
			}

		}	
    
    __syncthreads();

	  UInt l_data = 0;

#pragma unroll 64
		for (int i = 0; i < intprec; i++)
    {
			l_data += (UInt)((s_bit_cnt[i] >> tid) & 1u) << i;
    }

		__syncthreads();
		s_iblock[c_perm[tid]] = uint2int(l_data);
		__syncthreads();
		inv_xform(s_iblock);
		__syncthreads();

		//inv_cast
		result = dequantize<Int, Scalar>(1, s_emax[0]);
		result  *= (Scalar)(s_iblock[tid]);
    return result;
	}
}

template<class Scalar>
__global__
void
__launch_bounds__(64,5)
cudaDecode(Word *blocks,
           Scalar *out,
           const int3 dims,
           uint bsize)
{
  uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);

	extern __shared__ unsigned char smem[];

  const uint x_coord = threadIdx.x + blockIdx.x * 4;
  const uint y_coord = threadIdx.y + blockIdx.y * 4;
  const uint z_coord = threadIdx.z + blockIdx.z * 4;

  //const uint index = z_coord * gridDim.x * gridDim.y * blockDim.x * blockDim.y 
  //                 + y_coord * gridDim.x * blockDim.x 
  //                 + x_coord;
	Scalar val = decode<Scalar>(blocks + bsize*idx, smem, bsize);

  bool real_data = true;
  //
  // make sure we don't write out data that was padded out to make 
  // the block sizes all 4^3
  //
  if(x_coord >= dims.x || y_coord >= dims.y || z_coord >= dims.z)
  {
    real_data = false;
  }

  const uint out_index = z_coord * dims.x * dims.y 
                       + y_coord * dims.x 
                       + x_coord;
  if(real_data)
  {
    out[out_index] = val;
  }
  
	//inv_cast
}
template<class Scalar>
void decode(int3 dims, 
            thrust::device_vector<Word> &stream,
            Scalar *d_data,
            uint bsize)
{

  dim3 block_size = dim3(4, 4, 4);
  dim3 grid_size = dim3(dims.x, dims.y, dims.z);

  grid_size.x /= block_size.x; 
  grid_size.y /= block_size.y; 
  grid_size.z /= block_size.z;

  // Check to see if we need to increase the block sizes
  // in the case where dim[x] is not a multiple of 4
  if(dims.x % 4 != 0) grid_size.x++;
  if(dims.y % 4 != 0) grid_size.y++;
  if(dims.z % 4 != 0) grid_size.z++;

  const int some_magic_number = 64 * (8) + 4 + 4; 
  cudaDecode<Scalar> << < grid_size, block_size, some_magic_number >> >
    (raw_pointer_cast(stream.data()),
		 d_data,
     dims,
     bsize);
	cudaStreamSynchronize(0);
  //ec.chk("cudaInvXformCast");

  //  cudaEventRecord(stop, 0);
  //  cudaEventSynchronize(stop);
  //  cudaEventElapsedTime(&millisecs, start, stop);
  //ec.chk("cudadecode");
}

template<class Scalar>
void decode (int3 dims, 
             thrust::device_vector<Word > &block,
             thrust::device_vector<Scalar> &d_data,
             uint bsize)
{
	decode<Scalar>(dims, block, thrust::raw_pointer_cast(d_data.data()), bsize);
}

} // namespace cuZFP

#endif
