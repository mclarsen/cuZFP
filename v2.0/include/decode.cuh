#include <helper_math.h>
//dealing with doubles
#include "BitStream.cuh"
#define NBMASK 0xaaaaaaaaaaaaaaaaull
#define LDEXP(x, e) ldexp(x, e)

template<class Int, class Scalar>
__device__
Scalar
dequantize(Int x, int e)
{
	return LDEXP((double)x, e - (CHAR_BIT * c_sizeof_scalar - 2));
}

template<class Int, class Scalar, uint sizeof_scalar>
__host__
Scalar
dequantize(Int x, int e)
{
	return LDEXP((double)x, e - (CHAR_BIT * sizeof_scalar - 2));
}

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

template< class UInt, uint bsize>
__global__
void cudaDecode
(
UInt *q,
Bit<bsize> *stream,
const int *emax,
uint minbits, uint maxbits, uint maxprec, int minexp, unsigned long long group_count, uint size

)
{
	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	extern __shared__ Bit<bsize> sl_bits[];

	sl_bits[threadIdx.x] = stream[idx];

	decode_ints<UInt, bsize>(sl_bits[threadIdx.x], q + idx * bsize, minbits, maxbits, precision(emax[idx], maxprec, minexp), group_count, size);
	stream[idx] = sl_bits[threadIdx.x];
	//    encode_ints<UInt, bsize>(stream[idx], q + idx * bsize, minbits, maxbits, precision(emax[idx], maxprec, minexp), group_count, size);

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
read_bit(char &offset, uint *bits, Word *buffer, uint idx, const Word *begin)
{
	uint bit;
	if (!bits[idx]) {
		buffer[idx] = begin[offset++];
		bits[idx] = wsize;
	}
	bits[idx]--;
	bit = (uint)buffer[idx] & 1u;
	buffer[idx] >>= 1;
	return bit;
}
/* read 0 <= n <= 64 bits */
__host__ __device__
unsigned long long
read_bits(uint n, char &offset, uint *bits, Word *buffer, uint idx, const Word *begin)
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
		if (!bits[idx])
			value = begin[offset++];//*ptr++;
		else {
			value = buffer[idx];
			buffer[idx] = begin[offset++];//*ptr++;
			value += buffer[idx] << bits[idx];
			buffer[idx] >>= n - bits[idx];
		}
	}
	else {
		value = buffer[idx];
		if (bits[idx] < n) {
			/* not enough bits buffered; fetch wsize more */
			buffer[idx] = begin[offset++];//*ptr++;
			value += buffer[idx] << bits[idx];
			buffer[idx] >>= n - bits[idx];
			bits[idx] += wsize;
		}
		else
			buffer[idx] >>= n;
		value -= buffer[idx] << n;
		bits[idx] -= n;
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

const uint *idx_g,
const uint *idx_n,
uint *bits,
char *offset,
Word *buffer,

UInt *data,

const uint maxbits,
const uint intprec,
const uint kmin,
const unsigned long long orig_count,
const uint tid
)
{
	unsigned long long count = orig_count;
	uint new_bits = maxbits;

	for (uint k = intprec; k-- > kmin;){
		for (uint i = 0, m = idx_n[k], n = 0; i < idx_g[k] + 1; i++){
			if (new_bits){
				/* decode bit k for the next set of m values */
				m = MIN(m, new_bits);
				new_bits -= m;

				unsigned long long x = read_bits(m, offset[k], bits, buffer, k, stream.begin);
				x >>= tid - n;
				n += m;
				data[tid] += (UInt)(x & 1u) << k;

				/* continue with next bit plane if there are no more groups */
				if (!count || !new_bits)
					break;
				/* perform group test */
				new_bits--;
				uint test = read_bit(offset[k], bits, buffer, k, stream.begin);
				/* cache[k] with next bit plane if there are no more significant bits */
				if (!test || !new_bits)
					break;
				/* decode next group of m values */
				m = count & 0xfu;
				count >>= 4;
			}
		}
	}
}
template<class UInt, uint bsize>
__global__
void cudaDecodeBitstream
(
Bit<bsize> *stream,
const uint *idx_g,
const uint *idx_n,
const uint *bit_bits,
const char *bit_offset,
const Word *bit_buffer,

UInt *data,

const uint maxbits,
const uint intprec,
const uint kmin,
const unsigned long long orig_count
)
{
	extern __shared__ unsigned char smem[];

	UInt *s_data = (UInt*)smem;
	uint *s_idx_n = (uint*)&smem[64 * 8];
	uint *s_idx_g = (uint*)&smem[64 * 8 + 64 * 4];


	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bidx = idx*blockDim.x*blockDim.y*blockDim.z;
	char l_offset[64];
	uint l_bits[64];
	Word l_buffer[64];


	s_idx_n[tid] = idx_n[bidx + tid];
	s_idx_g[tid] = idx_g[bidx + tid];
	s_data[tid] = 0;
	
	for (int k = 0; k < 64; k++){
		l_offset[k] = bit_offset[bidx + k];
		l_bits[k] = bit_bits[bidx + k];
		l_buffer[k] = bit_buffer[bidx + k];
	}
	__syncthreads();
	decodeBitstream<UInt, bsize>(stream[idx], s_idx_g, s_idx_n, l_bits, l_offset, l_buffer, s_data, maxbits, intprec, kmin, orig_count, tid);
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
		}
	}
}

template<uint bsize>
__global__
void cudaDecodeGroup
(
Bit<bsize> *stream,
uint *idx_g,
uint *idx_n,
uint *bit_bits,
char *bit_offset,
Word *bit_buffer,

const uint maxbits,
const uint intprec,
const uint kmin,
const unsigned long long orig_count
)
{
	extern __shared__ unsigned char smem[];

	uint *s_idx_n = (uint*)&smem[0];
	uint *s_idx_g = (uint*)&smem[blockDim.x*blockDim.y*blockDim.z * sizeof(uint)];
	uint *s_bit_bits = (uint*)&smem[blockDim.x*blockDim.y*blockDim.z * 2*sizeof(uint)];
	char *s_bit_offset = (char*)&smem[blockDim.x*blockDim.y*blockDim.z * 3 * sizeof(uint)];
	Word *s_bit_buffer = (Word*)&smem[blockDim.x*blockDim.y*blockDim.z * (3 * sizeof(uint) + sizeof(char))];

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bidx = idx*blockDim.x*blockDim.y*blockDim.z;
	
	s_idx_g[tid] = 0;
	__syncthreads();

	if (tid == 0){
		insert_bit<bsize>(stream[idx], 
			s_idx_g,
			s_idx_n,
			s_bit_bits,
			s_bit_offset,
			s_bit_buffer,
			maxbits, intprec, kmin, orig_count);
	}
	__syncthreads();

	idx_g[bidx + tid] = s_idx_g[tid];
	idx_n[bidx + tid] = s_idx_n[tid];
	bit_bits[bidx + tid] = s_bit_bits[tid];
	bit_offset[bidx + tid] = s_bit_offset[tid];
	bit_buffer[bidx + tid] = s_bit_buffer[tid];

}
template<class UInt, uint bsize>
__global__
void cudaDecodePar
(
Bit<bsize> *stream,

UInt *data,

const uint maxbits,
const uint intprec,
const uint kmin,
const unsigned long long orig_count
)
{
	extern __shared__ unsigned char smem[];
	uint *s_idx_n = (uint*)&smem[0];
	uint *s_idx_g = (uint*)&smem[blockDim.x*blockDim.y*blockDim.z * sizeof(uint)];
	uint *s_bit_bits = (uint*)&smem[blockDim.x*blockDim.y*blockDim.z * 2 * sizeof(uint)];
	char *s_bit_offset = (char*)&smem[blockDim.x*blockDim.y*blockDim.z * 3 * sizeof(uint)];
	Word *s_bit_buffer = (Word*)&smem[blockDim.x*blockDim.y*blockDim.z * (3 * sizeof(uint) + sizeof(char))];

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bidx = idx*blockDim.x*blockDim.y*blockDim.z;

	s_idx_g[tid] = 0;
	__syncthreads();

	if (tid == 0){
		insert_bit<bsize>(stream[idx],
			s_idx_g,
			s_idx_n,
			s_bit_bits,
			s_bit_offset,
			s_bit_buffer,
			maxbits, intprec, kmin, orig_count);
	}__syncthreads();

	char l_offset[64];
	uint l_bits[64];
	Word l_buffer[64];

	for (int k = 0; k < 64; k++){
		l_offset[k] = s_bit_offset[k];
		l_bits[k] = s_bit_bits[k];
		l_buffer[k] = s_bit_buffer[k];
	}

	decodeBitstream<UInt, bsize>(stream[idx], s_idx_g, s_idx_n, l_bits, l_offset, l_buffer, data + bidx, maxbits, intprec, kmin, orig_count, tid);	
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

