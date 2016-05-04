#ifndef ENCODE_CUH
#define ENCODE_CUH

//#include <helper_math.h>
#include "shared.h"
#include "ull128.h"
#include "BitStream.cuh"
#include "WriteBitter.cuh"

#include <thrust/functional.h>


#define LDEXP(x, e) ldexp(x, e)
#define FREXP(x, e) frexp(x, e)
#define FABS(x) fabs(x)

const int ebias = 1023;

namespace cuZFP{

// map two's complement signed integer to negabinary unsigned integer
template<class Int, class UInt>
__device__ __host__
UInt int2uint(const Int x)
{
    return (x + (UInt)0xaaaaaaaaaaaaaaaaull) ^ (UInt)0xaaaaaaaaaaaaaaaaull;
}

template<typename Int, typename UInt>
struct int_2_uint : public thrust::unary_function<Int, UInt>
{
	__host__ __device__ Int operator()(const UInt &x) const
	{
		return (x + (UInt)0xaaaaaaaaaaaaaaaaull) ^ (UInt)0xaaaaaaaaaaaaaaaaull;
	}
};
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
    z = sz == 0 ? 0 : idx / (sz);
    uint rem = sz == 0 ? 0: idx % sz;
    y = sy == 0  ? 0  : rem / sy;
    x = sy == 0 ? 0 : rem % sy;
}

template<class Scalar>
__device__ __host__
int max_exp(const Scalar *p, uint mx, uint my, uint mz, uint sx, uint sy, uint sz)
{
//    uint mx,my,mz;
//    decompIdx(sx,sy,sz, idx, mx,my,mz);
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
void  fixed_point(Int *q, const Scalar *p, int emax, uint mx, uint my, uint mz, uint sx, uint sy, uint sz)
{
//    uint mx,my,mz;
//    decompIdx(sx,sy,sz, idx, mx,my,mz);

		//quantize
		//ASSUME CHAR_BIT is 8 and Scalar is 8
    Scalar w = LDEXP(1.0, intprec -2 -emax);
		uint i = 0;
    for (int z=mz; z<mz+4; z++)
        for (int y=my; y<my+4; y++)
            for (int x=mx; x<mx+4; x++,i++)
                q[i] =(Int)(p[z*sz+y*sy+x]*w);

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
fwd_xform_zy(Int* p)
{
    for (uint z = 0; z < 4; z++)
      for (uint y = 0; y < 4; y++)
        fwd_lift(p + 4 * y + 16 * z, 1);

}
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_xz(Int* p)
{
    for (uint x = 0; x < 4; x++)
      for (uint z = 0; z < 4; z++)
        fwd_lift(p + 16 * z + 1 * x, 4);

}
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_yx(Int* p)
{
    for (uint y = 0; y < 4; y++)
      for (uint x = 0; x < 4; x++)
        fwd_lift(p + 1 * x + 4 * y, 16);

}

// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform(Int* p)
{
    fwd_xform_zy(p);
    fwd_xform_xz(p);
    fwd_xform_yx(p);
}

template<class Int, class UInt>
__global__
void cudaint2uint
(
        const Int *p,
        UInt *q

        )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
    q[idx] = int2uint<Int, UInt>(p[c_perm[idx%64] + idx - idx % 64]);
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

template<class Int>
__global__
void cudaDecorrelateZY
(
        Int *p
        )
{
    int x = threadIdx.x + blockDim.x*blockIdx.x;
    int y = threadIdx.y  + blockDim.y*blockIdx.y;
    int z = threadIdx.z + blockDim.z*blockIdx.z;
    int idx = z*gridDim.x*blockDim.x*gridDim.y*blockDim.y + y*gridDim.x*blockDim.x + x;
    fwd_lift(p+4*idx,1);
}

template<class Int>
__global__
void cudaDecorrelateXZ
(
        Int *p
        )
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y  + blockDim.y*blockIdx.y;
    int k = threadIdx.z  + blockDim.z*blockIdx.z;

    int idx = j*gridDim.x*blockDim.x + i;
    fwd_lift(p + k%4 + 16*idx,4);
}

template<class Int>
__global__
void cudaDecorrelateYX
(
        Int *p
        )
{
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y  + blockDim.y*blockIdx.y;
    int k = threadIdx.z  + blockDim.z*blockIdx.z;

    int idx = j*gridDim.x*blockDim.x + i;
    fwd_lift(p + k % 16 + 64*idx, 16);
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
    //int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
    fixed_point(q + eidx*64, data, emax[eidx], x,y,z, 1, gridDim.x*blockDim.x*4, gridDim.x*blockDim.x*4*gridDim.y*blockDim.y*4);
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
    //int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
    emax[eidx] = max_exp(data, x,y,z, 1, gridDim.x*blockDim.x*4, gridDim.x*blockDim.x*4*gridDim.y*blockDim.y*4);

}


template<class Int, class Scalar>
__global__
void cudaEFPTransform
(
const Scalar *data,
Int *q
)
{
	int mx = threadIdx.x + blockDim.x*blockIdx.x;
	int my = threadIdx.y + blockDim.y*blockIdx.y;
	int mz = threadIdx.z + blockDim.z*blockIdx.z;
	int eidx = mz*gridDim.x*blockDim.x*gridDim.y*blockDim.y + my*gridDim.x*blockDim.x + mx;
	extern __shared__ long long sh_q[];


	mx *= 4; my *= 4; mz *= 4;
	//int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
	int emax = max_exp(data, mx, my, mz, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);

	uint sz = gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4;
	uint sy = gridDim.x*blockDim.x * 4;
	uint sx = 1;
	fixed_point(sh_q +(threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64, data, emax, mx, my, mz, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);
	for (int i = 0; i < 64; i++){
		q[eidx * 64 + i] = sh_q[(threadIdx.x+threadIdx.y*4+threadIdx.z*16) * 64 + i];
	}
}

template<class Int, class Scalar>
__global__
void cudaEFPDTransform
(
const Scalar *data,
Int *q
)
{
	int mx = threadIdx.x + blockDim.x*blockIdx.x;
	int my = threadIdx.y + blockDim.y*blockIdx.y;
	int mz = threadIdx.z + blockDim.z*blockIdx.z;
	int eidx = mz*gridDim.x*blockDim.x*gridDim.y*blockDim.y + my*gridDim.x*blockDim.x + mx;
	extern __shared__ long long sh_q[];


	mx *= 4; my *= 4; mz *= 4;
	//int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
	int emax = max_exp(data, mx, my, mz, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);

	uint sz = gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4;
	uint sy = gridDim.x*blockDim.x * 4;
	uint sx = 1;
	fixed_point(sh_q + (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64, data, emax, mx, my, mz, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);
	fwd_xform(sh_q + (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64);
	for (int i = 0; i < 64; i++){
		q[eidx * 64 + i] = sh_q[(threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64 + i];
	}
}


template<class Int, class UInt, class Scalar, uint bsize>
__global__
void cudaEFPDI2UTransform
(
const Scalar *data,
UInt *p,
Bit<bsize> *stream

)
{
	int mx = threadIdx.x + blockDim.x*blockIdx.x;
	int my = threadIdx.y + blockDim.y*blockIdx.y;
	int mz = threadIdx.z + blockDim.z*blockIdx.z;
	int eidx = mz*gridDim.x*blockDim.x*gridDim.y*blockDim.y + my*gridDim.x*blockDim.x + mx;
	extern __shared__ long long sh_q[];


	mx *= 4; my *= 4; mz *= 4;
	//int idx = z*gridDim.x*gridDim.y*blockDim.x*blockDim.y*16 + y*gridDim.x*blockDim.x*4+ x;
	int emax = max_exp(data, mx, my, mz, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);
	
	stream[eidx].emax = emax;
//	uint sz = gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4;
//	uint sy = gridDim.x*blockDim.x * 4;
//	uint sx = 1;
	fixed_point(sh_q + (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64, data, emax, mx, my, mz, 1, gridDim.x*blockDim.x * 4, gridDim.x*blockDim.x * 4 * gridDim.y*blockDim.y * 4);
	fwd_xform(sh_q + (threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64);


	//fwd_order
	for (int i = 0; i < 64; i++){
		uint idx = eidx * 64 + i;
		p[idx] = int2uint<Int, UInt>(sh_q[(threadIdx.x + threadIdx.y * 4 + threadIdx.z * 16) * 64 + c_perm[i]]);
	}

}

inline
__device__ __host__
void
encodeBitplane
(
unsigned long long count,

unsigned long long x,
const unsigned char g,
unsigned char h,
const unsigned char *g_cnt,

//uint &h, uint &n_cnt, unsigned long long &cnt,
Bitter &bitters,
unsigned char &sbits

)
{
	unsigned long long cnt = count;
	cnt >>= h * 4;
	uint n_cnt = g_cnt[h];

	/* serial: output one bit plane at a time from MSB to LSB */

	sbits = 0;
	/* encode bit k for first n values */
	x = write_bitters(bitters, make_bitter(x, 0), n_cnt, sbits);
	while (h++ < g) {
		/* output a one bit for a positive group test */
		write_bitter(bitters, make_bitter(1, 0), sbits);
		/* add next group of m values to significant set */
		uint m = cnt & 0xfu;
		cnt >>= 4;
		n_cnt += m;
		/* encode next group of m values */
		x = write_bitters(bitters, make_bitter(x, 0), m, sbits);
	}
	/* if there are more groups, output a zero bit for a negative group test */
	if (cnt) {
		write_bitter(bitters, make_bitter(0, 0), sbits);
	}
}

template<class UInt, uint bsize>
__global__
void cudaEncodeUInt
(
const unsigned long long count,
uint size,
const UInt* data,
const unsigned char *g_cnt,
Bit<bsize> *stream
)
{
//	int mx = threadIdx.x + blockDim.x*blockIdx.x;
//	int my = threadIdx.y + blockDim.y*blockIdx.y;
//	int mz = threadIdx.z + blockDim.z*blockIdx.z;
//	int eidx = mz*gridDim.x*blockDim.x*gridDim.y*blockDim.y + my*gridDim.x*blockDim.x + mx;

	extern __shared__ unsigned char smem[];
	__shared__ unsigned char *sh_g, *sh_sbits; 
	__shared__ Bitter *sh_bitters;
	__shared__ uint *s_emax_bits;
	sh_g = &smem[0];
	sh_sbits = &smem[64];
	sh_bitters = (Bitter*)&smem[64 + 64];
	s_emax_bits = (uint*)&smem[64 + 64 + 16 * 64];

	unsigned long long x;

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;

	uint bidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*blockDim.x*blockDim.y*blockDim.z;

	Bitter bitter = make_bitter(0, 0);
	unsigned char sbit = 0;
	uint kmin = 0;

//	uint k = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid == 0){
		s_emax_bits[0] = 1;
		int emax = stream[bidx / 64].emax;
		int maxprec = precision(emax, c_maxprec, c_minexp);
		kmin = intprec > maxprec ? intprec - maxprec : 0;

		uint e = maxprec ? emax + ebias : 0;
		if (e){
			//write_bitters(bitter[0], make_bitter(2 * e + 1, 0), ebits, sbit[0]);
			stream[bidx / 64].begin[0] = 2 * e + 1;
			s_emax_bits[0] = c_ebits + 1;
		}
	}
	__syncthreads();

	/* extract bit plane k to x[k] */
	unsigned long long y = 0;
	for (uint i = 0; i < size; i++)
		y += ((data[bidx + i] >> tid) & (unsigned long long)1) << i;
	x = y;

	__syncthreads();
	/* count number of positive group tests g[k] among 3*d in d dimensions */
	sh_g[tid] = 0;
	for (unsigned long long c = count; y; y >>= c & 0xfu, c >>= 4)
		sh_g[tid]++;

	__syncthreads();
	if (tid == 0){
		unsigned char cur = sh_g[intprec - 1];

		for (int i = intprec - 1; i-- > kmin;) {
			if (cur < sh_g[i])
				cur = sh_g[i];
			else if (cur > sh_g[i])
				sh_g[i] = cur;
		}
	}

	__syncthreads();
	//	g[k] = sh_g[threadIdx.x];

	unsigned char g = sh_g[tid];
	unsigned char h = sh_g[min(tid + 1, intprec - 1)];


	encodeBitplane(count, x, g, h, g_cnt, bitter, sbit);
	sh_bitters[63 - tid] = bitter;
	sh_sbits[63 - tid] = sbit;

	__syncthreads();
	if (tid == 0){
		uint tot_sbits = s_emax_bits[0];// sbits[0];
		uint offset = 0;
		for (int i = 0; i < intprec; i++){
			if (sh_sbits[i] <= 64){
				write_outx(sh_bitters, stream[bidx / 64].begin, tot_sbits, offset, i, sh_sbits[i]);
			}
			else{
				write_outx(sh_bitters, stream[bidx / 64].begin, tot_sbits, offset, i, 64);
				write_outy(sh_bitters, stream[bidx / 64].begin, tot_sbits, offset, i, sh_sbits[i] - 64);
			}
		}
	}
}

template<class Int, class UInt, class Scalar, uint bsize>
void encode
(
int nx, int ny, int nz,
const Scalar *d_data,
thrust::device_vector<cuZFP::Bit<bsize> > &stream,
thrust::device_vector<int> &emax,
    const uint maxprec,
    const unsigned long long group_count,
    const uint size
)
{
  thrust::device_vector<UInt> buffer(nx*ny*nz);
  thrust::device_vector<unsigned char> d_g_cnt;

  dim3 emax_size(nx / 4, ny / 4, nz / 4);

  dim3 block_size(8, 8, 8);
  dim3 grid_size = emax_size;
  grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  //ErrorCheck ec;

  stream.resize(emax_size.x * emax_size.y * emax_size.z);
  emax.resize(emax_size.x * emax_size.y * emax_size.z);

  //  cudaEvent_t start, stop;
  //  float millisecs;

  //  cudaEventCreate(&start);
  //  cudaEventCreate(&stop);
  //  cudaEventRecord(start, 0);


  //ec.chk("pre-cudaMaxExp");
  cuZFP::cudaMaxExp << <grid_size, block_size >> >
    (
    thrust::raw_pointer_cast(emax.data()),
    d_data
    );
  cudaStreamSynchronize(0);
  //ec.chk("cudaMaxExp");

  //for (int i = 0; i < emax.size(); i++){
  //  std::cout << emax[i] << " ";
  //  if (!(i % nx))
  //    std::cout << std::endl;
  //  if (!(i % nx*ny))
  //    std::cout << std::endl;
  //}
  block_size = dim3(4, 4, 4);
  grid_size = emax_size;
  grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;
  //ec.chk("pre-cudaEFPDI2UTransform");
  cuZFP::cudaEFPDI2UTransform <Int, UInt, Scalar> << < grid_size, block_size, sizeof(Int) * 4 * 4 * 4 * 4 * 4 * 4 >> >
    (
    d_data,
    thrust::raw_pointer_cast(buffer.data())
    );
  //ec.chk("post-cudaEFPDI2UTransform");





  unsigned long long count = group_count;
  thrust::host_vector<unsigned char> g_cnt(10);
  uint sum = 0;
  g_cnt[0] = 0;
  for (int i = 1; i < 10; i++){
    sum += count & 0xf;
    g_cnt[i] = sum;
    count >>= 4;
  }
  d_g_cnt = g_cnt;

  block_size = dim3(4, 4, 4);
  grid_size = dim3(nx, ny, nz);
  grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

  cuZFP::cudaEncodeUInt<UInt, bsize> << <grid_size, block_size, (2 * sizeof(unsigned char) + sizeof(Bitter)) * 64 >> >
    (
    kmin, group_count, size,
    thrust::raw_pointer_cast(buffer.data()),
    thrust::raw_pointer_cast(d_g_cnt.data()),
    thrust::raw_pointer_cast(stream.data())
    );
  cudaStreamSynchronize(0);
  //ec.chk("cudaEncode");
  //  cudaEventRecord(stop, 0);
  //  cudaEventSynchronize(stop);
  //  cudaEventElapsedTime(&millisecs, start, stop);


  //  cout << "encode GPU in time: " << millisecs << endl;

}

template<class Int, class UInt, class Scalar, uint bsize>
void encode
(
int nx, int ny, int nz,
thrust::device_vector<Scalar> &d_data,
thrust::device_vector<cuZFP::Bit<bsize> > &stream,
thrust::device_vector<int> &emax,
const uint maxprec,
const unsigned long long group_count,
const uint size
)
{
	encode<Int, UInt, Scalar, bsize>(nx, ny, nz,
		thrust::raw_pointer_cast(d_data.data()),
		stream,
		emax,
		maxprec,
		group_count,
		size);
}

template<class Int, class UInt, class Scalar, uint bsize>
void encode
(
int nx, int ny, int nz,
const thrust::host_vector<Scalar> &h_data,
thrust::device_vector<cuZFP::Bit<bsize> > &stream,
thrust::device_vector<int> &emax,
const uint maxprec,
const unsigned long long group_count,
const uint size
)
{
	thrust::device_vector<UInt> buffer(nx*ny*nz);
	thrust::device_vector<unsigned char> d_g_cnt;

	thrust::device_vector<Scalar> d_data = h_data;

	encode<Int, UInt, Scalar, bsize>(nx, ny, nz, d_data, stream, emax, maxprec, group_count, size);
}

}
#endif
