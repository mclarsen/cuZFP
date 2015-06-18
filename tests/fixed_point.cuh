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

template<class Scalar>
__device__ __host__
int max_exp(const Scalar *p, uint idx, uint sx, uint sy, uint sz)
{
    uint mz = idx / (sz);
    uint rem = idx % sz;
    uint my = rem == 0  ? 0  : rem / sy;
    uint mx = rem == 0 ? 0 : rem % sy;
    Scalar fmax = 0;
    for (int z=mz; z<mz+4; z++)
        for (int y=my; y<my+4; y++)
            for (int x=mx; x<mx+4; x++)
                fmax = MAX(fmax, FABS(p[z*sz+y*sy+x]));

    return exponent(fmax);
}


//gather from p into q
template<class Int, class Scalar>
void  fwd_cast(Int *q, const Scalar *p, int emax, uint idx, uint sx, uint sy, uint sz)
{
    uint mz = idx / (sz);
    uint rem = idx % sz;
    uint my = rem == 0  ? 0  : sy / rem;
    uint mx = rem == 0 ? 0 : sy % rem;



    Scalar w = LDEXP(1, intprec -2 -emax);
    for (int z=mz; z<4; z++)
        for (int y=my; y<4; y++)
            for (int x=mx; x<4; x++,q++)
                *q =(Int)(p[z*sz+y*sy+x]*w);

}
