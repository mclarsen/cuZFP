#ifndef DECODE_CUH
#define DECODE_CUH

//#include <helper_math.h>
//dealing with doubles
#include "BitStream.cuh"
#include <thrust/device_vector.h>
#define NBMASK 0xaaaaaaaaaaaaaaaaull
#define LDEXP(x, e) ldexp(x, e)

namespace cuZFP{

#ifdef __CUDA_ARCH__
template<class Int, class Scalar>
__device__
Scalar
dequantize(Int x, int e)
{
	return LDEXP((double)x, e - (CHAR_BIT * c_sizeof_scalar - 2));
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
/* inverse block-floating-point transform from signed integers */
template<class Int, class Scalar>
__host__ __device__
void
inv_cast(const Int* p, Scalar* q, int emax, uint mx, uint my, uint mz, uint sx, uint sy, uint sz)
{
	Scalar s;
#ifndef __CUDA_ARCH__
	s = dequantize<Int, Scalar, sizeof(Scalar)>(1, emax);
#else
	/* compute power-of-two scale factor s */
	s = dequantize<Int, Scalar>(1, emax);
#endif
	/* compute p-bit float x = s*y where |y| <= 2^(p-2) - 1 */
	//  do
	//    *fblock++ = (Scalar)(s * *iblock++);
	//  while (--n);
	for (int z = mz; z < mz + 4; z++)
		for (int y = my; y < my + 4; y++)
			for (int x = mx; x < mx + 4; x++, p++)
				q[z*sz + y*sy + x*sx] = (Scalar)(s * *p);

}

/* inverse lifting transform of 4-vector */
template<class Int>
__host__ __device__
static void
inv_lift(Int* p, uint s)
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
__host__ __device__
static void
inv_xform_yx(Int* p)
{
	uint x, y;
	for (y = 0; y < 4; y++)
		for (x = 0; x < 4; x++)
			inv_lift(p + 1 * x + 4 * y, 16);

}

/* transform along y */
template<class Int>
__host__ __device__
static void
inv_xform_xz(Int* p)
{
	uint x, z;
	for (x = 0; x < 4; x++)
		for (z = 0; z < 4; z++)
			inv_lift(p + 16 * z + 1 * x, 4);

}

/* transform along x */
template<class Int>
__host__ __device__
static void
inv_xform_zy(Int* p)
{
	uint y, z;
	for (z = 0; z < 4; z++)
		for (y = 0; y < 4; y++)
			inv_lift(p + 4 * y + 16 * z, 1);

}

/* inverse decorrelating 3D transform */
template<class Int>
__host__ __device__
static void
inv_xform(Int* p)
{

	inv_xform_yx(p);
	inv_xform_xz(p);
	inv_xform_zy(p);
}

/* map two's complement signed integer to negabinary unsigned integer */
template<class Int, class UInt>
__host__ __device__
Int
uint2int(UInt x)
{
	return (x ^ NBMASK) - NBMASK;
}




/* decompress sequence of unsigned integers */
template<class Int, class UInt>
static uint
decode_ints_old(BitStream* stream, uint minbits, uint maxbits, uint maxprec, UInt* data, uint size, unsigned long long count)
{
  BitStream s = *stream;
  uint intprec = CHAR_BIT * (uint)sizeof(UInt);
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n, test;
  unsigned long long x;

  /* initialize data array to all zeros */
  for (i = 0; i < size; i++)
    data[i] = 0;

  /* input one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; k-- > kmin;) {
    /* decode bit plane k */
    UInt* p = data;
    for (m = n;;) {
      /* decode bit k for the next set of m values */
      m = MIN(m, bits);
      bits -= m;
      for (x = stream->read_bits(m); m; m--, x >>= 1)
        *p++ += (UInt)(x & 1u) << k;
      /* continue with next bit plane if there are no more groups */
      if (!count || !bits)
        break;
      /* perform group test */
      bits--;
      test = stream->read_bit();
      /* continue with next bit plane if there are no more significant bits */
      if (!test || !bits)
        break;
      /* decode next group of m values */
      m = count & 0xfu;
      count >>= 4;
      n += m;
    }
    /* exit if there are no more bits to read */
    if (!bits)
      goto exit;
  }

  /* read at least minbits bits */
  while (bits > maxbits - minbits) {
    bits--;
    stream->read_bit();
  }

exit:
  *stream = s;
  return maxbits - bits;
}

template<uint bsize>
__global__
void cudaRewind
(
Bit<bsize> * stream
)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;

	stream[idx].rewind();
}

template<class UInt, uint bsize>
__device__ __host__
uint
decode_ints(Bit<bsize> & stream, UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long count, uint size)
{
	uint intprec = CHAR_BIT * (uint)sizeof(UInt);
	uint kmin = intprec > maxprec ? intprec - maxprec : 0;
	uint bits = maxbits;
	uint i, k, m, n, test;
	unsigned long long x;

	/* initialize data array to all zeros */
	for (i = 0; i < size; i++)
		data[i] = 0;

	/* input one bit plane at a time from MSB to LSB */
	for (k = intprec, n = 0; k-- > kmin;) {
		/* decode bit plane k */
		UInt* p = data;
		for (m = n;;) {
			if (bits){
				/* decode bit k for the next set of m values */
				m = MIN(m, bits);
				bits -= m;
				for (x = stream.read_bits(m); m; m--, x >>= 1)
					*p++ += (UInt)(x & 1u) << k;
				/* continue with next bit plane if there are no more groups */
				if (!count || !bits)
					break;
				/* perform group test */
				bits--;
				test = stream.read_bit();
				/* continue with next bit plane if there are no more significant bits */
				if (!test || !bits)
					break;
				/* decode next group of m values */
				m = count & 0xfu;
				count >>= 4;
				n += m;
			}
		}
	}

	/* read at least minbits bits */
	while (bits > maxbits - minbits) {
		bits--;
		stream.read_bit();
	}

	return maxbits - bits;

}

template<class Int, class UInt>
__global__
void cudaInvOrder
(
const UInt *p,
Int *q
)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
	q[c_perm[idx % 64] + idx - idx % 64] = uint2int<Int, UInt>(p[idx]);

}

template<class Int>
__global__
void cudaInvXForm
(
Int *iblock
)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
	inv_xform(iblock + idx * 64);

}


template<class Int>
__global__
void cudaInvXFormYX
(
Int *iblock
)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;
	int k = threadIdx.z + blockDim.z*blockIdx.z;
	int idx = j*gridDim.x*blockDim.x + i;
	inv_lift(iblock + k % 16 + 64 * idx, 16);

}


template<class Int>
__global__
void cudaInvXFormXZ
(
Int *iblock
)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	int j = threadIdx.y + blockDim.y*blockIdx.y;
	int k = threadIdx.z + blockDim.z*blockIdx.z;
	int idx = j*gridDim.x*blockDim.x + i;
	inv_lift(iblock + k % 4 + 16 * idx, 4);

}

template<class Int>
__global__
void cudaInvXFormZY
(
Int *p
)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
	inv_lift(p + 4 * idx, 1);
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
	unsigned long long value;
	/* because shifts by 64 are not possible, treat n = 64 specially */
	if (n == bitsize(value)) {
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


//__shared__ uint s_bits[64 * 64];
//__shared__ Word s_buffer[64 * 64];

//__shared__ uint s_idx_n[64];
//__shared__ uint s_idx_g[64];

template<class UInt, uint bsize>
__host__ __device__
void decodeBitstream
(
const Bit<bsize> &stream,

uint idx_g,
uint idx_n,
unsigned long long count,
uint new_bits,
uint bits,
char offset,
Word buffer,

UInt &data,

const uint tid,
const uint k
)
{


	for (uint i = 0, m = idx_n, n = 0; i < idx_g + 1; i++){
		/* decode bit k for the next set of m values */
		m = MIN(m, new_bits);
		new_bits -= m;

		unsigned long long x = read_bits(m, offset, bits, buffer, stream.begin);
		x >>= tid - n;
		n += m;
		data += (UInt)(x & 1u) << k;

		/* continue with next bit plane if there are no more groups */
		//if (!count || !new_bits)
		//	break;
		/* perform group test */
		new_bits--;
		uint test = read_bit(offset, bits, buffer, stream.begin);
		/* cache[k] with next bit plane if there are no more significant bits */
		//if (!test || !new_bits)
		//	break;
		/* decode next group of m values */
		m = count & 0xfu;
		count >>= 4;
	}
}

template<class UInt, uint bsize, uint num_sidx>
__global__
void cudaDecodeBitstream
(
const size_t *sidx,
Bit<bsize> *stream,
const uint *idx_g,
const uint *idx_n,
const uint *bit_bits,
const char *bit_offset,
const Word *bit_buffer,
const unsigned long long *bit_cnt,
const uint *bit_rmn_bits,

UInt *data,

const uint maxbits,
const uint intprec,
const uint kmin,
const unsigned long long orig_count
)
{

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bidx = idx*blockDim.x*blockDim.y*blockDim.z;

	extern __shared__ unsigned char smem[];
#if 1
	size_t *s_sidx = (size_t*)&smem[0];
	if (tid < num_sidx)
		s_sidx[tid] = sidx[tid];
	__syncthreads();

  UInt *s_data = (UInt*)&smem[s_sidx[0]];
	uint *s_idx_n = (uint*)&smem[s_sidx[1]];
	uint *s_idx_g = (uint*)&smem[s_sidx[2]];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[s_sidx[3]];
	uint *s_bit_rmn_bits = (uint*)&smem[s_sidx[4]];

#else
	UInt *s_data = (UInt*)smem;
	uint *s_idx_n = (uint*)&smem[64 * 8];
	uint *s_idx_g = (uint*)&smem[64 * 8 + 64 * 4];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[64 * 8 + 64 * 4 + 64 * 4];
	uint *s_bit_rmn_bits = (uint*)&smem[64 * 8 + 64 * 4 + 64 * 4 + 64 * 8];
#endif
	s_bit_rmn_bits[tid] = bit_rmn_bits[bidx + tid];
	s_bit_cnt[tid] = bit_cnt[bidx + tid];
	s_idx_n[tid] = idx_n[bidx + tid];
	s_idx_g[tid] = idx_g[bidx + tid];
	s_data[tid] = 0;
	
	char l_offset[64];
	uint l_bits[64];
	Word l_buffer[64];
	for (int k = 0; k < 64; k++){
		l_offset[k] = bit_offset[bidx + k];
		l_bits[k] = bit_bits[bidx + k];
		l_buffer[k] = bit_buffer[bidx + k];
	}
	__syncthreads();
	for (uint k = kmin; k < intprec; k++){
//		unsigned long long count = bit_cnt[k];
//		uint new_bits = bit_rmn_bits[k];

		decodeBitstream<UInt, bsize>(
			stream[idx],
			s_idx_g[k],
			s_idx_n[k],
			s_bit_cnt[k],
			s_bit_rmn_bits[k],
			l_bits[k],
			l_offset[k],
			l_buffer[k],
			s_data[tid],
			maxbits, intprec, tid, k);
	}
	data[bidx + tid] = s_data[tid];
}


template<uint bsize>
__device__ __host__
void insert_bit
(
Bit<bsize> &stream,
uint *idx_g,
uint *idx_n,
uint *bit_bits,
char *bit_offset,
Word *bit_buffer,
unsigned long long *bit_cnt,
uint *bit_rmn_bits,

const uint maxbits,
const uint intprec,
const uint kmin,
const unsigned long long orig_count
)
{
	uint bits = maxbits;
	unsigned long long count = orig_count;

	/* input one bit plane at a time from MSB to LSB */
	for (uint k = intprec, n = 0; k-- > kmin;) {
		/* decode bit plane k */
		idx_n[k] = n;
		bit_rmn_bits[k] = bits;
		bit_cnt[k] = count;

		bit_bits[k] = stream.bits;
		bit_offset[k] = stream.offset;
		bit_buffer[k] = stream.buffer;
		for (uint m = n;; idx_g[k]++) {

			if (bits){
				/* decode bit k for the next set of m values */
				m = MIN(m, bits);
				bits -= m;
				unsigned long long x = stream.read_bits(m);
				/* continue with next bit plane if there are no more groups */
				if (!count || !bits){

					break;
				}
				/* perform group test */
				bits--;
				uint test = stream.read_bit();
				/* continue with next bit plane if there are no more significant bits */
				if (!test || !bits){
					break;
				}
				/* decode next group of m values */
				m = count & 0xfu;
				count >>= 4;
				n += m;
			}
			else
				break;
		}
	}
}

template<uint bsize, uint num_sidx>
__global__
void cudaDecodeGroup
(
const size_t *sidx,
Bit<bsize> *stream,
uint *idx_g,
uint *idx_n,
uint *bit_bits,
char *bit_offset,
Word *bit_buffer,
unsigned long long *bit_cnt,
uint *bit_rmn_bits,

const uint maxbits,
const uint intprec,
const uint kmin,
const unsigned long long orig_count
)
{
	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bidx = idx*blockDim.x*blockDim.y*blockDim.z;

	//uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	//uint bidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*blockDim.x*blockDim.y*blockDim.z;
	//uint idx = tid + bidx;

	extern __shared__ unsigned char smem[];
#if 1
	size_t *s_sidx = (size_t*)&smem[0];
	if (tid < num_sidx)
		s_sidx[tid] = sidx[tid];
	__syncthreads();
	uint *s_idx_n = (uint*)&smem[s_sidx[0]];
	uint *s_idx_g = (uint*)&smem[s_sidx[1]];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[s_sidx[2]];
	uint *s_bit_rmn_bits = (uint*)&smem[s_sidx[3]];
	char *s_bit_offset = (char*)&smem[s_sidx[4]];
	uint *s_bit_bits = (uint*)&smem[s_sidx[5]];
	Word *s_bit_buffer = (Word*)&smem[s_sidx[6]];

#else
	uint *s_idx_n = (uint*)&smem[64 * 8];
	uint *s_idx_g = (uint*)&smem[64 * 8 + 64 * 4];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[64 * 8 + 64 * 4 + 64 * 4];
	uint *s_bit_rmn_bits = (uint*)&smem[64 * 8 + 64 * 4 + 64 * 4 + 64 * 8];
	char *s_bit_offset = (char*)&smem[64 * 8 + 64 * 4 + 64 * 4 + 64 * 8 + 64 * 4];
	uint *s_bit_bits = (uint*)&smem[64 * 8 + 64 * 4 + 64 * 4 + 64 * 8 + 64 * 4 + 64];
	Word *s_bit_buffer = (Word*)&smem[64 * 8 + 64 * 4 + 64 * 4 + 64 * 8 + 64 * 4 + 64 + 64 * 4];
#endif
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
	}

	__syncthreads();
	idx_g[bidx + tid] = s_idx_g[tid];
	idx_n[bidx + tid] = s_idx_n[tid];
	bit_bits[bidx + tid] = s_bit_bits[tid];
	bit_offset[bidx + tid] = s_bit_offset[tid];
	bit_buffer[bidx + tid] = s_bit_buffer[tid];
	bit_cnt[bidx + tid] = s_bit_cnt[tid];
	bit_rmn_bits[bidx + tid] = s_bit_rmn_bits[tid];


}
template<class UInt, uint bsize, uint num_sidx>
__global__
void cudaDecodePar
(
const size_t *sidx,
Bit<bsize> *stream,

UInt *data,

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
#if 1
	size_t *s_sidx = (size_t*)&smem[0];
	if (tid < num_sidx)
		s_sidx[tid] = sidx[tid];
	__syncthreads();
	uint *s_idx_n = (uint*)&smem[s_sidx[0]];
	uint *s_idx_g = (uint*)&smem[s_sidx[1]];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[s_sidx[2]];
	uint *s_bit_rmn_bits = (uint*)&smem[s_sidx[3]];
	char *s_bit_offset = (char*)&smem[s_sidx[4]];
	uint *s_bit_bits = (uint*)&smem[s_sidx[5]];
	Word *s_bit_buffer = (Word*)&smem[s_sidx[6]];
	UInt *s_data = (UInt*)&smem[s_sidx[7]];
#else
	uint *s_idx_n = (uint*)&smem[0];
	uint *s_idx_g = (uint*)&smem[64 * sizeof(uint)];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[64 * (4+4)];
	uint *s_bit_rmn_bits = (uint*)&smem[64 *( 4 + 4 + 8)];
	char *s_bit_offset = (char*)&smem[64 *(4 + 4 + 8 + 4)];
	uint *s_bit_bits = (uint*)&smem[64 *(4 + 4 + 8 + 4 + 1)];
	Word *s_bit_buffer = (Word*)&smem[64 *(4 + 4 + 8 +4 + 1 + 4)];
	UInt *s_data = (UInt*)&smem[64 * (4 + 4 + 8 + 4 + 1 + 4 + 8)];
#endif
	s_idx_g[tid] = 0;
	s_data[tid] = 0;
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
			s_data[tid],
			tid, k);
	}

	data[bidx + tid] = s_data[tid];
}
template<class Int, class Scalar>
__global__
void cudaInvCast
(
const int *emax,
Scalar *data,
const Int *q
)
{
	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	int z = threadIdx.z + blockDim.z*blockIdx.z;
	int eidx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;

	x *= 4; y *= 4; z *= 4;
	//int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
	inv_cast<Int, Scalar>(q + eidx * 64, data, emax[eidx], x, y, z, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);
}

template<class Int, class Scalar>
__global__
void cudaInvXformCast
(
const int *emax,
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
	inv_cast<Int, Scalar>(iblock + eidx * 64, data, emax[eidx], x, y, z, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);

}

template<class Int, class UInt, uint bsize, uint num_sidx>
__global__
void cudaDecodeInvOrder
(
size_t *sidx,
Bit<bsize> *stream,

Int *data,

const uint maxbits,
const uint intprec,
const unsigned long long orig_count

)
{
	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bdim = blockDim.x*blockDim.y*blockDim.z;
	uint bidx = idx*bdim;

	extern __shared__ unsigned char smem[];
#if 1
	size_t *s_sidx = (size_t*)&smem[0];
	if (tid < num_sidx)
		s_sidx[tid] = sidx[tid];
	__syncthreads();
	uint *s_idx_n = (uint*)&smem[s_sidx[0]];
	uint *s_idx_g = (uint*)&smem[s_sidx[1]];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[s_sidx[2]];
	uint *s_bit_rmn_bits = (uint*)&smem[s_sidx[3]];
	char *s_bit_offset = (char*)&smem[s_sidx[4]];
	uint *s_bit_bits = (uint*)&smem[s_sidx[5]];
	Word *s_bit_buffer = (Word*)&smem[s_sidx[6]];
	UInt *s_data = (UInt*)&smem[s_sidx[7]];
	uint *s_kmin = (uint*)&smem[s_sidx[8]];
#else
	uint *s_idx_n = (uint*)&smem[0];
	uint *s_idx_g = (uint*)&smem[64 * 4];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[64 * (4 + 4)];
	uint *s_bit_rmn_bits = (uint*)&smem[64 * (4 + 4 + 8)];
	char *s_bit_offset = (char*)&smem[64 * (4 + 4 + 8 + 4)];
	uint *s_bit_bits = (uint*)&smem[64 * (4 + 4 + 8 + 4 + 1)];
	Word *s_bit_buffer = (Word*)&smem[64 * (4 + 4 + 8 + 4 + 1 + 4)];
	UInt *s_data = (UInt*)&smem[64 * (4 + 4 + 8 + 4 + 1 + 4 + 8)];
#endif
	s_idx_g[tid] = 0;
	s_data[tid] = 0;
	__syncthreads();

	if (tid == 0){
		stream[idx].read_bit();
		uint ebits = c_ebits + 1;
		int emax = stream[idx].read_bits(ebits - 1) - ebias;
		int maxprec = precision(emax, c_maxprec, c_minexp);
		s_kmin[0] = intprec > maxprec ? intprec - maxprec : 0;

		insert_bit<bsize>(
			stream[idx],
			s_idx_g,
			s_idx_n,
			s_bit_bits,
			s_bit_offset,
			s_bit_buffer,
			s_bit_cnt,
			s_bit_rmn_bits,
			maxbits, intprec, s_kmin[0], orig_count);
	}	__syncthreads();

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

	data[c_perm[tid] + bidx] = uint2int<Int, UInt>(s_data[tid]);

}

template<class Int, class UInt, class Scalar, uint bsize, uint num_sidx>
__global__
void cudaDecode
(
size_t *sidx,
Bit<bsize> *stream,

Scalar *out,

const uint intprec,
const unsigned long long orig_count

)
{
	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bdim = blockDim.x*blockDim.y*blockDim.z;
	uint bidx = idx*bdim;

	extern __shared__ unsigned char smem[];
#if 1
	size_t *s_sidx = (size_t*)&smem[0];
	if (tid < num_sidx)
		s_sidx[tid] = sidx[tid];
	__syncthreads();
	uint *s_idx_n = (uint*)&smem[s_sidx[0]];
	uint *s_idx_g = (uint*)&smem[s_sidx[1]];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[s_sidx[2]];
	uint *s_bit_rmn_bits = (uint*)&smem[s_sidx[3]];
	char *s_bit_offset = (char*)&smem[s_sidx[4]];
	uint *s_bit_bits = (uint*)&smem[s_sidx[5]];
	Word *s_bit_buffer = (Word*)&smem[s_sidx[6]];
	UInt *s_data = (UInt*)&smem[s_sidx[7]];
	Int *s_iblock = (Int*)&smem[s_sidx[8]];
	uint *s_kmin = (uint*)&smem[s_sidx[9]];

#else
	uint *s_idx_n = (uint*)&smem[0];
	uint *s_idx_g = (uint*)&smem[64 * 4];
	unsigned long long *s_bit_cnt = (unsigned long long*)&smem[64 * (4 + 4)];
	uint *s_bit_rmn_bits = (uint*)&smem[64 * (4 + 4 + 8)];
	char *s_bit_offset = (char*)&smem[64 * (4 + 4 + 8 + 4)];
	uint *s_bit_bits = (uint*)&smem[64 * (4 + 4 + 8 + 4 + 1)];
	Word *s_bit_buffer = (Word*)&smem[64 * (4 + 4 + 8 + 4 + 1 + 4)];
	UInt *s_data = (UInt*)&smem[64 * (4 + 4 + 8 + 4 + 1 + 4 + 8)];
#endif
	s_idx_g[tid] = 0;
	s_data[tid] = 0;
	s_bit_rmn_bits[tid] = 0;
	__syncthreads();

	int emax = 0;
	if (tid == 0){
		stream[idx].read_bit();
		uint ebits = c_ebits + 1;
		emax = stream[idx].read_bits(ebits - 1) - ebias;
		int maxprec = precision(emax, c_maxprec, c_minexp);
		s_kmin[0] = intprec > maxprec ? intprec - maxprec : 0;

		insert_bit<bsize>(
			stream[idx],
			s_idx_g,
			s_idx_n,
			s_bit_bits,
			s_bit_offset,
			s_bit_buffer,
			s_bit_cnt,
			s_bit_rmn_bits,
			c_maxbits, intprec, s_kmin[0], orig_count);
	}	__syncthreads();

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

	s_iblock[c_perm[tid]] = uint2int<Int, UInt>(s_data[tid]);
	__syncthreads();
	if (tid == 0){
		uint mx = blockIdx.x, my = blockIdx.y, mz = blockIdx.z;
		mx *= 4; my *= 4; mz *= 4;
		inv_xform(s_iblock);
		inv_cast<Int, Scalar>(s_iblock, out, emax, mx, my, mz, 1, gridDim.x*blockDim.x, gridDim.x*blockDim.x * gridDim.y*blockDim.y);


	}
}
template<class Int, class UInt, class Scalar, uint bsize>
void decode
(
int nx, int ny, int nz,
thrust::device_vector<cuZFP::Bit<bsize> > &stream,
thrust::device_vector<int> &emax,
Scalar *d_data,
    uint maxprec,
    unsigned long long group_count,
    uint maxbits
)
{
  //ErrorCheck ec;
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  dim3 emax_size(nx / 4, ny / 4, nz / 4);

  thrust::device_vector<Int> q(nx*ny*nz);
  dim3 block_size = dim3(8, 8, 8);
  dim3 grid_size = emax_size;
  grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
  cuZFP::cudaRewind<bsize> << < grid_size, block_size >> >
    (
    raw_pointer_cast(stream.data())
    );
  //ec.chk("cudaRewind");


  block_size = dim3(4, 4, 4);
  grid_size = dim3(nx, ny, nz);
  grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
  size_t blcksize = block_size.x *block_size.y * block_size.z;
  size_t s_idx[9] = { sizeof(size_t) * 9, blcksize * sizeof(uint), blcksize * sizeof(uint), +blcksize * sizeof(unsigned long long), blcksize * sizeof(uint), blcksize * sizeof(char), blcksize * sizeof(uint), blcksize * sizeof(Word), blcksize * sizeof(UInt) };
  thrust::inclusive_scan(s_idx, s_idx + 9, s_idx);
  const size_t shmem_size = thrust::reduce(s_idx, s_idx + 9);
  thrust::device_vector<size_t> d_sidx(s_idx, s_idx + 9);
  //cudaDecodeInvOrder<Int, UInt, bsize, 9> << < grid_size, block_size, shmem_size>> >
  cuZFP::cudaDecodeInvOrder<Int, UInt, bsize, 9> << < grid_size, block_size, 64 * (4 + 4 * 8 + 4 + 1 + 4 + 8 + 8) >> >

    (
    raw_pointer_cast(d_sidx.data()),
    raw_pointer_cast(stream.data()),
    raw_pointer_cast(q.data()),
    maxbits,
    intprec,
    kmin,
    group_count);
  cudaStreamSynchronize(0);
  //ec.chk("cudaDecodeInvOrder");
  //  for (int i=0; i<q.size(); i++){
  //    std::cout << q[i] << " ";
  //    if (!(i % nx))
  //      std::cout << std::endl;
  //    if (!(i%nx*ny))
  //      std::cout << std::endl;
  //  }
  //  std::cout << std:: endl;

  //  cudaEventRecord(stop, 0);
  //  cudaEventSynchronize(stop);
  //  cudaEventElapsedTime(&millisecs, start, stop);
  //ec.chk("cudadecode");
  //  cout << "decode parallel GPU in time: " << millisecs << endl;

  block_size = dim3(4, 4, 4);
  grid_size = emax_size;
  grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
  cuZFP::cudaInvXformCast<Int, Scalar> << <grid_size, block_size >> >(
    thrust::raw_pointer_cast(emax.data()),
    d_data,
    thrust::raw_pointer_cast(q.data()));
  cudaStreamSynchronize(0);
  //ec.chk("cudaInvXformCast");

  //  cudaEventRecord(stop, 0);
  //  cudaEventSynchronize(stop);
  //  cudaEventElapsedTime(&millisecs, start, stop);
  //ec.chk("cudadecode");
}
template<class Int, class UInt, class Scalar, uint bsize>
void decode
(
int nx, int ny, int nz,
thrust::device_vector<cuZFP::Bit<bsize> > &stream,
thrust::device_vector<int> &emax,
thrust::device_vector<Scalar> &d_data,
uint maxprec,
unsigned long long group_count,
uint maxbits
)
{
	decode<Int, UInt, Scalar, bsize>(
		nx, ny, nz, 
		stream, 
		emax, 
		thrust::raw_pointer_cast(d_data.data()),
		maxprec, group_count,
		maxbits);
}
}

#endif
