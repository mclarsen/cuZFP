#include <limits.h>
#include <math.h>
#include "../inline/inline.h"
#include "../inline/bitstream.c"

/* private functions ------------------------------------------------------- */

/* return normalized floating-point exponent for x >= 0 */
static int
_t1(exponent, Scalar)(Scalar x)
{
  if (x > 0) {
    int e;
    FREXP(x, &e);
    /* clamp exponent in case x is denormal */
    return MAX(e, 1 - EBIAS);
  }
  return -EBIAS;
}

/* compute maximum exponent in block of n values */
static int
_t1(exponent_block, Scalar)(const Scalar* p, uint n)
{
  Scalar max = 0;
  do {
    Scalar f = FABS(*p++);
    if (max < f)
      max = f;
  } while (--n);
  return _t1(exponent, Scalar)(max);
}

/* map floating-point number x to integer relative to exponent e */
static Scalar
_t1(quantize, Scalar)(Scalar x, int e)
{
  return LDEXP(x, (CHAR_BIT * (int)sizeof(Scalar) - 2) - e);
}

/* forward block-floating-point transform to signed integers */
static void
_t1(fwd_cast, Scalar)(Int* iblock, const Scalar* fblock, uint n, int emax)
{
  /* compute power-of-two scale factor s */
  Scalar s = _t1(quantize, Scalar)(1, emax);
  /* compute p-bit int y = s*x where x is floating and |y| <= 2^(p-2) - 1 */
  do
    *iblock++ = (Int)(s * *fblock++);
  while (--n);
}

/* forward lifting transform of 4-vector */
static void
_t1(fwd_lift, Int)(Int* p, uint s)
{
  Int x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  /*
  ** non-orthogonal transform
  **        ( 4  4  4  4) (x)
  ** 1/16 * ( 5  1 -1 -5) (y)
  **        (-4  4  4 -4) (z)
  **        (-2  6 -6  2) (w)
  */
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

/* map two's complement signed integer to negabinary unsigned integer */
static UInt
_t1(int2uint, Int)(Int x)
{
  return (x + NBMASK) ^ NBMASK;
}

/* reorder signed coefficients and convert to unsigned integer */
static void
_t1(fwd_order, Int)(UInt* ublock, const Int* iblock, const uchar* perm, uint n)
{
  do
    *ublock++ = _t1(int2uint, Int)(iblock[*perm++]);
  while (--n);
}

/* compress sequence of unsigned integers */
static uint
_t1(encode_ints, UInt)(BitStream* _restrict stream, uint minbits, uint maxbits, uint maxprec, const UInt* data, uint size, uint64 count)
{
  BitStream s = *stream;
  uint intprec = CHAR_BIT * (uint)sizeof(UInt);
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint i, k, m, n;
  uint64 x;

  if (!maxbits)
    return 0;

  /* output one bit plane at a time from MSB to LSB */
  for (k = intprec, n = 0; k-- > kmin;) {
    /* extract bit plane k to x */
    x = 0;
    for (i = 0; i < size; i++)
      x += ((data[i] >> k) & (uint64)1) << i;
    /* encode bit plane */
    for (m = n;;) {
      /* encode bit k for next set of m values */
      m = MIN(m, bits);
      x = stream_write_bits(&s, x, m);
      bits -= m;
      /* continue with next bit plane if there are no more groups */
      if (!count || !bits)
        break;
      /* perform group test on remaining bits of x */
      stream_write_bit(&s, !!x);
      bits--;
      /* continue with next bit plane if there are no more significant bits */
      if (!x || !bits)
        break;
      /* encode next group of m values */
      m = count & 0xfu;
      count >>= 4;
      n += m;
    }
    /* exit if there are no more bits to write */
    if (!bits)
      goto exit;
  }

  /* write at least minbits bits by padding with zeros */
  while (bits > maxbits - minbits) {
    stream_write_bit(&s, 0);
    bits--;
  }

exit:
  *stream = s;
  return maxbits - bits;
}

/* public functions -------------------------------------------------------- */

/* encode block of integers */
void
_t2(zfp_encode_block, Int, DIMS)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, Int* iblock)
{
  _cache_align(UInt ublock[BLOCK_SIZE]);
  /* perform decorrelating transform */
  _t2(fwd_xform, Int, DIMS)(iblock);
  /* reorder signed coefficients and convert to unsigned integer */
  _t1(fwd_order, Int)(ublock, iblock, PERM, BLOCK_SIZE);
  /* encode integer coefficients */
  _t1(encode_ints, UInt)(stream, minbits, maxbits, maxprec, ublock, BLOCK_SIZE, GROUP_SIZE);
}

// maximum number of bit planes to encode
static uint
precision(int maxexp, uint maxprec, int minexp)
{
	return MIN(maxprec, MAX(0, maxexp - minexp + 8));
}

/* encode contiguous floating-point block */
void
_t2(zfp_encode_block, Scalar, DIMS)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const Scalar* fblock)
{
  _cache_align(Int iblock[BLOCK_SIZE]);
  /* compute maximum exponent */
  int emax = _t1(exponent_block, Scalar)(fblock, BLOCK_SIZE);
  /* perform forward block-floating-point transform */
  _t1(fwd_cast, Scalar)(iblock, fblock, BLOCK_SIZE, emax);
  /* encode common exponent */
	uint e = maxprec ? emax + EBIAS : 0;
	int ebits = EBITS + 1;
  stream_write_bits(stream, 2*e+1, EBITS + 1);
  /* encode integer block */
	_t2(zfp_encode_block, Int, DIMS)(stream, minbits - ebits, maxbits - ebits, _t2(precision, Scalar, DIMS)(emax, maxprec, minexp), iblock);
}
