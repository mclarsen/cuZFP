#include "codec.h"

// map two's complement signed integer to negabinary unsigned integer
static UInt
int2uint(Int x)
{
  return (x + (UInt)0xaaaaaaaaaaaaaaaaull) ^ (UInt)0xaaaaaaaaaaaaaaaaull;
}

// map floating-point number x to integer relative to exponent e
static Int
float2int(Scalar x, int e)
{
  return (Int)LDEXP(x, intprec - 2 - e);
}

// block-floating-point transform to signed integers
static int
fwd_cast(Int* q, const Scalar* p, uint sx, uint sy, uint sz)
{
  // compute maximum exponent
  Scalar fmax = 0;
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx)
        fmax = MAX(fmax, FABS(*p));
  p -= 4 * sz;
  int emax = exponent(fmax);

  // normalize by maximum exponent and convert to fixed-point
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx, q++)
        *q = float2int(*p, emax);

  return emax;
}

// lifting transform of 4-vector
static void
fwd_lift(Int* p, uint s)
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
static void
fwd_xform(Int* p)
{
  for (uint z = 0; z < 4; z++)
    for (uint y = 0; y < 4; y++)
      fwd_lift(p + 4 * y + 16 * z, 1);
  for (uint x = 0; x < 4; x++)
    for (uint z = 0; z < 4; z++)
      fwd_lift(p + 16 * z + 1 * x, 4);
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      fwd_lift(p + 1 * x + 4 * y, 16);
}

static void
encode_ints(BitStream* stream, const UInt* data, uint minbits, uint maxbits, uint maxprec, uint64 count, uint size)
{
  if (!maxbits)
    return;

  uint bits = maxbits;
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;

  // output one bit plane at a time from MSB to LSB
  for (uint k = intprec, n = 0; k-- > kmin;) {
    // extract bit plane k to x
    uint64 x = 0;
    for (uint i = 0; i < size; i++)
      x += ((data[i] >> k) & (uint64)1) << i;
    // encode bit plane
    for (uint m = n;; m = count & 0xfu, count >>= 4, n += m) {
      // encode bit k for next set of m values
      m = MIN(m, bits);
      bits -= m;
      x = stream_write_bits(stream, x, m);
      // exit if there are no more bits to write
      if (!bits)
        return;
      // continue with next bit plane if out of groups or group test passes
      if (!count || (bits--, stream_write_bit(stream, !!x), !x))
        break;
    }
  }

  // pad with zeros in case fewer than minbits bits have been written
  while (bits-- > maxbits - minbits)
    stream_write_bit(stream, 0);
}

// encode 4*4*4 block from p using strides
void
encode3(BitStream* stream, const Scalar* p, uint sx, uint sy, uint sz, uint minbits, uint maxbits, uint maxprec, int minexp)
{
  // perform block-floating-point transform
  Int q[64];
  int emax = fwd_cast(q, p, sx, sy, sz);
  // perform decorrelating transform
  fwd_xform(q);
  // reorder coefficients and convert to unsigned integer
  UInt buffer[64];
  for (uint i = 0; i < 64; i++)
    buffer[i] = int2uint(q[perm[i]]);
  // encode block
  stream_write_bits(stream, emax + ebias, ebits);
  encode_ints(stream, buffer, minbits, maxbits, precision(emax, maxprec, minexp), 0x46acca631ull, 64);
}
