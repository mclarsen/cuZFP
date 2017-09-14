#ifndef SHARED_H
#define SHARED_H

#include <type_info.cuh>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

#define LDEXP(x, e) ldexp(x, e)
#define FREXP(x, e) frexp(x, e)
#define FABS(x) fabs(x)

typedef unsigned long long Word;

static const uint wsize = bitsize(Word);

__constant__ unsigned char c_perm_1[4];
__constant__ unsigned char c_perm_2[16];
__constant__ unsigned char c_perm[64];

namespace cuZFP
{
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
    return MAX(e, 1 - get_ebias<Scalar>());
  }
  return -get_ebias<Scalar>();
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

template<typename Scalar>
Scalar
inline __device__
quantize_factor(const int &exponent, Scalar);

template<>
float
inline __device__
quantize_factor<float>(const int &exponent, float)
{
	return  LDEXP(1.0, get_precision<float>() - 2 - exponent);
}

template<>
double
inline __device__
quantize_factor<double>(const int &exponent, double)
{
	return  LDEXP(1.0, get_precision<double>() - 2 - exponent);
}



} // namespace cuZFP
#endif
