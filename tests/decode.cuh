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
	s = dequantize<Int, Scalar, sizeof(uint)>(1, emax);
#else
	/* compute power-of-two scale factor s */
	s = dequantize<Int, Scalar>(1, emax);
#endif
	/* compute p-bit float x = s*y where |y| <= 2^(p-2) - 1 */
//  do
//    *fblock++ = (Scalar)(s * *iblock++);
//  while (--n);
  for (int z=mz; z<mz+4; z++)
      for (int y=my; y<my+4; y++)
          for (int x=mx; x<mx+4; x++,p++)
              q[z*sz+y*sy+x*sx] = (Scalar)(s * *p);

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

/* inverse decorrelating 3D transform */
template<class Int>
__host__ __device__
static void
inv_xform(Int* p)
{
  uint x, y, z;
  /* transform along z */
  for (y = 0; y < 4; y++)
    for (x = 0; x < 4; x++)
      inv_lift(p + 1 * x + 4 * y, 16);
  /* transform along y */
  for (x = 0; x < 4; x++)
    for (z = 0; z < 4; z++)
      inv_lift(p + 16 * z + 1 * x, 4);
  /* transform along x */
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      inv_lift(p + 4 * y + 16 * z, 1);
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
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
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
    extern __shared__ Bit<bsize> s_bits[];

    s_bits[threadIdx.x] = stream[idx];

    decode_ints<UInt, bsize>(s_bits[threadIdx.x], q + idx * bsize, minbits, maxbits, precision(emax[idx], maxprec, minexp), group_count, size);
    stream[idx] = s_bits[threadIdx.x];
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
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
    q[c_perm[idx%64] + idx - idx % 64] = uint2int<Int, UInt>(p[idx]);

}

template<class Int>
__global__
void cudaInvXForm
(
        Int *iblock
        )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
    inv_xform(iblock + idx*64);

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
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int eidx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;

    x *= 4; y*=4; z*=4;
    //int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
    inv_cast<Int, Scalar>(q + eidx*64, data, emax[eidx], x,y,z, 1, gridDim.x*blockDim.x*4, gridDim.x*blockDim.x*4*gridDim.y*blockDim.y*4);
}

