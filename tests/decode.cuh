#include <helper_math.h>
//dealing with doubles
#include "BitStream.cuh"
#define NBMASK 0xaaaaaaaaaaaaaaaaull
#define LDEXP(x, e) ldexp(x, e)

template<class Int, class Scalar, int sizeof_scalar>
__host__ __device__
Scalar
dequantize(Int x, int e)
{
  return LDEXP((double)x, e - (CHAR_BIT * sizeof_scalar - 2));
}

/* inverse block-floating-point transform from signed integers */
template<class Int, class Scalar, int sizeof_scalar>
__host__ __device__
void
inv_cast(const Int* iblock, Scalar* fblock, uint n, int emax)
{
  /* compute power-of-two scale factor s */
  Scalar s = dequantize<Int, Scalar, sizeof_scalar>(1, emax);
  /* compute p-bit float x = s*y where |y| <= 2^(p-2) - 1 */
  do
    *fblock++ = (Scalar)(s * *iblock++);
  while (--n);
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
static Int
uint2int(UInt x)
{
  return (x ^ NBMASK) - NBMASK;
}


/* reorder unsigned coefficients and convert to signed integer */
template<class Int, class UInt>
__host__ __device__
static void
inv_order(const UInt* ublock, Int* iblock, const unsigned char* perm, uint n)
{
  do
    iblock[*perm++] = uint2int<UInt>(*ublock++);
  while (--n);
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

template<class UInt, uint bsize>
__device__ __host__
static uint
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
