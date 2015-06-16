#ifndef CODEC_H
#define CODEC_H

#include <math.h>
#include "types.h"
#include "bitstream.h"

#if 0
// single precision
#define FABS(x) fabsf(x)
#define FREXP(x, e) frexpf(x, e)
#define LDEXP(x, e) ldexpf(x, e)
typedef float Scalar;
typedef uint32 UInt;
typedef int32 Int;
enum {
  intprec = 32,
  ebits = 8,
  ebias = 127
};
#else
// double precision
#define FABS(x) fabs(x)
#define FREXP(x, e) frexp(x, e)
#define LDEXP(x, e) ldexp(x, e)
typedef double Scalar;
typedef uint64 UInt;
typedef int64 Int;
enum {
  intprec = 64,
  ebits = 11,
  ebias = 1023
};
#endif

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

static const uchar
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

// return normalized floating-point exponent for x >= 0
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

// 3D specific functions

// maximum number of bit planes to encode
static uint
precision(int maxexp, uint maxprec, int minexp)
{
  return MIN(maxprec, MAX(0, maxexp - minexp + 8));
}

// encode a 3D block of scalars
void encode3(BitStream* stream, const Scalar* p, uint sx, uint sy, uint sz, uint minbits, uint maxbits, uint maxprec, int minexp);

#endif
