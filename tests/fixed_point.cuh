#include <helper_math.h>

#define LDEXP(x, e) ldexp(x, e)
#define FREXP(x, e) frexp(x, e)
#define FABS(x) fabs(x)

const int intprec = 64;
const int ebias = 1023;

// return normalized floating-point exponent for x >= 0
template<class Scalar>
__host__ __device__
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

template<class T, bool mult_only>
__device__ __host__
void setLDEXP
(
    uint idx,
        const T *in,
        T *out,
        const T w,
        const int exp
        )
{
    if (mult_only){
        out[idx] = in[idx] * w;
    }
    else
        out[idx] = LDEXP(in[idx], exp);
}


template<class T>
__host__ __device__
void setFREXP
    (
        uint idx,
        const T *in,
        T *out,
        int *nptr
        )
{
    out[idx] = FREXP(in[idx], &nptr[ idx] );
}

// block-floating-point transform to signed integers
template<class Int, class Scalar>
int fwd_cast(Int* q, const Scalar* p, uint sx, uint sy, uint sz)
{
  // compute maximum exponent
  Scalar fmax = 0;
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx)
        fmax = MAX(fmax, FABS(*p));
  p -= 4 * sz;
  int emax = exponent(fmax);

  double w = LDEXP(1, intprec -2 -emax);
  // normalize by maximum exponent and convert to fixed-point
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx, q++){
          *q =(Int)(*p*w);
      }

  return emax;
}

__host__ __device__
void decompIdx
(
        uint sx,
        uint sy,
        uint sz,
        int idx,
        uint &x,
        uint &y,
        uint &z
        )
{
    z = idx / (sz);
    uint rem = idx % sz;
    y = rem == 0  ? 0  : rem / sy;
    x = rem == 0 ? 0 : rem % sy;
}

template<class Scalar>
__device__ __host__
int max_exp(const Scalar *p, uint idx, uint sx, uint sy, uint sz)
{
    uint mx,my,mz;
    decompIdx(sx,sy,sz, idx, mx,my,mz);
    Scalar fmax = 0;
    for (int z=mz; z<mz+4; z++)
        for (int y=my; y<my+4; y++)
            for (int x=mx; x<mx+4; x++)
                fmax = MAX(fmax, FABS(p[z*sz+y*sy+x]));

    return exponent(fmax);
}


//gather from p into q
template<class Int, class Scalar>
__host__  __device__
void  fixed_point(Int *q, const Scalar *p, int emax, uint idx, uint sx, uint sy, uint sz)
{
    uint mx,my,mz;
    decompIdx(sx,sy,sz, idx, mx,my,mz);


    Scalar w = LDEXP(1.0, intprec -2 -emax);
    for (int z=mz; z<mz+4; z++)
        for (int y=my; y<my+4; y++)
            for (int x=mx; x<mx+4; x++,q++)
                *q =(Int)(p[z*sz+y*sy+x]*w);

}


// lifting transform of 4-vector
template <class Int>
__device__ __host__
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
template<class Int>
__device__ __host__
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


template<class Int>
__global__
void cudaDecorrelate
(
        Int *p
        )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
    fwd_xform(p + idx*64);
}

template<class Int, class Scalar>
__global__
void cudaFixedPoint
(
        const int *emax,
        const Scalar *data,
        Int *q
        )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int eidx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;

    x *= 4; y*=4; z*=4;
    int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
    fixed_point(q + eidx*64, data, emax[eidx], idx, 1, gridDim.x*blockDim.x*4, gridDim.x*blockDim.x*4*gridDim.y*blockDim.y*4);
}

template<class Scalar>
__global__
void cudaMaxExp
(
        int *emax,
    Scalar *data
        )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int eidx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;

    x *= 4; y*=4; z*=4;
    int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
    emax[eidx] = max_exp(data, idx, 1, gridDim.x*blockDim.x*4, gridDim.x*blockDim.x*4*gridDim.y*blockDim.y*4);

}
