#include <limits.h>
#include <math.h>
#include "../inline/inline.h"
#include "../inline/bitstream.c"

/* private functions ------------------------------------------------------- */

/* map integer x relative to exponent e to floating-point number */
static Scalar
_t1(dequantize, Scalar)(Int x, int e)
{
  return LDEXP(x, e - (CHAR_BIT * (int)sizeof(Scalar) - 2));
}

/* inverse block-floating-point transform from signed integers */
static void
_t1(inv_cast, Scalar)(const Int* iblock, Scalar* fblock, uint n, int emax)
{
  /* compute power-of-two scale factor s */
  Scalar s = _t1(dequantize, Scalar)(1, emax);
  /* compute p-bit float x = s*y where |y| <= 2^(p-2) - 1 */
  do
    *fblock++ = (Scalar)(s * *iblock++);
  while (--n);
}

/* inverse lifting transform of 4-vector */
static void
_t1(inv_lift, Int)(Int* p, uint s)
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

/* map two's complement signed integer to negabinary unsigned integer */
static Int
_t1(uint2int, UInt)(UInt x)
{
  return (x ^ NBMASK) - NBMASK;
}

/* reorder unsigned coefficients and convert to signed integer */
static void
_t1(inv_order, Int)(const UInt* ublock, Int* iblock, const uchar* perm, uint n)
{
  do
    iblock[*perm++] = _t1(uint2int, UInt)(*ublock++);
  while (--n);
}

/* decompress sequence of unsigned integers */
static uint
_t1(decode_ints, UInt)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, UInt* data, uint size, uint64 count)
{
  BitStream s = *stream;
  uint intprec = CHAR_BIT * (uint)sizeof(UInt);
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n, test;
  uint64 x;

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
      for (x = stream_read_bits(&s, m); m; m--, x >>= 1)
        *p++ += (UInt)(x & 1u) << k;
      /* continue with next bit plane if there are no more groups */
      if (!count || !bits)
        break;
      /* perform group test */
      bits--;
      test = stream_read_bit(&s);
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
    stream_read_bit(&s);
  }

exit:
  *stream = s;
  return maxbits - bits;
}

/* public functions -------------------------------------------------------- */

/* decode block of integers */
void
_t2(zfp_decode_block, Int, DIMS)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, Int* iblock)
{
  _cache_align(UInt ublock[BLOCK_SIZE]);
  /* decode integer coefficients */
  _t1(decode_ints, UInt)(stream, minbits, maxbits, maxprec, ublock, BLOCK_SIZE, GROUP_SIZE);
  /* reorder unsigned coefficients and convert to signed integer */
  _t1(inv_order, Int)(ublock, iblock, PERM, BLOCK_SIZE);
  /* perform decorrelating transform */
  _t2(inv_xform, Int, DIMS)(iblock);
}

/* decode contiguous floating-point block */
void
_t2(zfp_decode_block, Scalar, DIMS)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, Scalar* fblock)
{
  _cache_align(Int iblock[BLOCK_SIZE]);
  /* decode common exponent */
  int emax = stream_read_bits(stream, EBITS) - EBIAS;
  /* decode integer block */
  _t2(zfp_decode_block, Int, DIMS)(stream, minbits - EBITS, maxbits - EBITS, _t2(precision, Scalar, DIMS)(emax, maxprec, minexp), iblock);
  /* perform inverse block-floating-point transform */
  _t1(inv_cast, Scalar)(iblock, fblock, BLOCK_SIZE, emax);
}
