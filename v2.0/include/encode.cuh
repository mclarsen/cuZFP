#ifndef ENCODE_CUH
#define ENCODE_CUH

//#include <helper_math.h>
#include "shared.h"
#include "ull128.h"
#include "BitStream.cuh"
#include "WriteBitter.cuh"
#include "shared.h"
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
template<class Int, class Scalar, int intprec>
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
int max_exp_block(const Scalar *p, uint mx, uint my, uint mz, uint sx, uint sy, uint sz)
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
template<class Int, class Scalar, int intprec>
__host__  __device__
void  fixed_point_block(Int *q, const Scalar *p, int emax, uint mx, uint my, uint mz, uint sx, uint sy, uint sz)
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

template<class Scalar>
__device__ __host__
int max_exp_flat(const Scalar *p, uint begin_idx, uint end_idx)
{
  //    uint mx,my,mz;
  //    decompIdx(sx,sy,sz, idx, mx,my,mz);
  Scalar fmax = 0;
  for (int i = begin_idx; i < end_idx; i++)
    fmax = MAX(fmax, FABS(p[i]));

  return exponent(fmax);
}

//gather from p into q
template<class Int, class Scalar, int intprec>
__host__  __device__
void  fixed_point_flat(Int *q, const Scalar *p, int emax, uint begin_idx, uint end_idx)
{
  //    uint mx,my,mz;
  //    decompIdx(sx,sy,sz, idx, mx,my,mz);

  //quantize
  //ASSUME CHAR_BIT is 8 and Scalar is 8
  Scalar w = LDEXP(1.0, intprec - 2 - emax);

  for (uint i = begin_idx; i < end_idx; i++)
        q[i] = (Int)(p[i] * w);

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
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_zy(Int* p)
{

	fwd_lift<Int,1>(p + 4 * threadIdx.x + 16 * threadIdx.z);

	//for (uint z = 0; z < 4; z++)
	//	for (uint y = 4; y-- > 0;)
 //       fwd_lift<Int, 1>(p + 4 * y + 16 * z);

}
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_xz(Int* p)
{
	fwd_lift<Int, 4>(p + 16 * threadIdx.z + 1 * threadIdx.x);
	//for (uint x = 4; x-- > 0;)
    //  for (uint z = 4; z-- > 0;)
				//fwd_lift<Int, 4>(p + 16 * z + 1 * x);

}
// forward decorrelating transform
template<class Int>
__device__ __host__
static void
fwd_xform_yx(Int* p)
{
	fwd_lift<Int, 16>(p + 1 * threadIdx.x + 4 * threadIdx.z);
	//for (uint y = 4; y-- > 0;)
 //     for (uint x = 4; x-- > 0;)
	//			fwd_lift<Int, 16>(p + 1 * x + 4 * y);

}

// forward decorrelating transform
template<class Int>
__device__ 
static void
fwd_xform(Int* p)
{
  fwd_xform_zy(p);
	__syncthreads();
	fwd_xform_xz(p);
	__syncthreads();
	fwd_xform_yx(p);
//	if (tid == 0){
////    fwd_xform_zy(p, tid);
//		fwd_xform_yx(p);
//
//	}
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
    fixed_point_block(q + eidx*64, data, emax[eidx], x,y,z, 1, gridDim.x*blockDim.x*4, gridDim.x*blockDim.x*4*gridDim.y*blockDim.y*4);
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
    emax[eidx] = max_exp_block(data, x,y,z, 1, gridDim.x*blockDim.x*4, gridDim.x*blockDim.x*4*gridDim.y*blockDim.y*4);

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

template<typename Int, typename UInt, typename Scalar, uint bsize, int intprec>
__device__
void encode
(
	const Scalar *data,
	const uint size,
	unsigned char *smem,

	unsigned char * &sh_sbits,
	Bitter * &sh_bitters,
	uint *&s_emax_bits,
	Word *blocks

)
{
	__shared__ unsigned char *sh_g;
	__shared__ Scalar *sh_data, *sh_reduce;
	__shared__ int *sh_emax;
	__shared__ Int *sh_q;
	__shared__ UInt *sh_p;
	__shared__ uint *sh_m, *sh_n;

	sh_g = &smem[0];
	sh_sbits = &smem[64];
	sh_bitters = (Bitter*)&smem[64 + 64];
	sh_p = (UInt*)&smem[64 + 64 + 16 * 64];
	sh_data = (Scalar*)&sh_p[64];
	sh_reduce = (Scalar*)&sh_data[64];
	sh_q = (Int*)&sh_reduce[32];
	sh_m = (uint*)&sh_q[64];
	sh_n = (uint*)&sh_m[64];
	s_emax_bits = (uint*)&sh_n[64];
	sh_emax = (int*)&s_emax_bits[1];
	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);

	Bitter bitter = make_bitter(0, 0);
	unsigned char sbit = 0;
	uint kmin = 0;

	sh_data[tid] = data[(threadIdx.z + blockIdx.z * 4)*gridDim.x * gridDim.y * blockDim.x * blockDim.y + (threadIdx.y + blockIdx.y * 4)*gridDim.x * blockDim.x + (threadIdx.x + blockIdx.x * 4)];

	__syncthreads();
	//max_exp
	if (tid < 32)
		sh_reduce[tid] = max(sh_data[tid], sh_data[tid + 32]);
	if (tid < 16)
		sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + 16]);
	if (tid < 8)
		sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + 8]);
	if (tid < 4)
		sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + 4]);
	if (tid < 2)
		sh_reduce[tid] = max(sh_reduce[tid], sh_reduce[tid + 2]);
	if (tid == 0){
		sh_reduce[0] = max(sh_reduce[tid], sh_reduce[tid + 1]);
		sh_emax[0] = exponent(sh_reduce[0]);
	}
	__syncthreads();
	//if (tid == 0){
	//	//uint mx = blockIdx.x, my = blockIdx.y, mz = blockIdx.z;
	//	//mx *= 4; my *= 4; mz *= 4;
	//	//sh_emax[0] = max_exp_block(data, mx, my, mz, 1, gridDim.x * blockDim.x, gridDim.x * gridDim.y * blockDim.x * blockDim.y);

	//	//fixed_point_block<Int, Scalar>(sh_q, data, sh_emax[0], mx, my, mz, 1, gridDim.x * blockDim.x, gridDim.x * gridDim.y * blockDim.x * blockDim.y);
	//}
	//__syncthreads();

	//fixed_point
	Scalar w = LDEXP(1.0, intprec - 2 - sh_emax[0]);
	sh_q[tid] = (Int)(sh_data[tid] * w);


	fwd_xform(sh_q);
	__syncthreads();
	//fwd_order
	sh_p[tid] = int2uint<Int, UInt>(sh_q[c_perm[tid]]);
	if (tid == 0){
		s_emax_bits[0] = 1;
		int maxprec = precision(sh_emax[0], c_maxprec, c_minexp);
		kmin = intprec > maxprec ? intprec - maxprec : 0;

		uint e = maxprec ? sh_emax[0] + ebias : 0;
		if (e){
			//write_bitters(bitter[0], make_bitter(2 * e + 1, 0), ebits, sbit[0]);
			blocks[idx * bsize] = 2 * e + 1;
			s_emax_bits[0] = c_ebits + 1;
		}
	}
	__syncthreads();

	/* extract bit plane k to x[k] */
	unsigned long long y = 0;
	for (uint i = 0; i < size; i++)
		y += ((sh_p[i] >> tid) & (unsigned long long)1) << i;

	unsigned long long x = y;

	__syncthreads();
	sh_m[tid] = 0;
	sh_n[tid] = 0;

	//temporarily use sh_n as a buffer
	uint *sh_test = sh_n;
	for (int i = 0; i < 64; i++){
		if (!!(x >> i))
			sh_n[tid] = i + 1;
	}

	if (tid < 63){
		sh_m[tid] = sh_n[tid + 1];
	}

	__syncthreads();
	if (tid == 0){
		for (int i = intprec - 1; i-- > 0;){
			if (sh_m[i] < sh_m[i + 1])
				sh_m[i] = sh_m[i + 1];
		}
	}
	__syncthreads();

	int bits = 128;
	int n = 0;
	/* step 2: encode first n bits of bit plane */
	bits -= sh_m[tid];
	x >>= sh_m[tid];
	x = (sh_m[tid] != 64) * x;
	n = sh_m[tid];
	/* step 3: unary run-length encode remainder of bit plane */
	for (; n < size && bits && (bits--, !!x); x >>= 1, n++)
		for (; n < size - 1 && bits && (bits--, !(x & 1u)); x >>= 1, n++)
			;
	__syncthreads();

	bits = (128 - bits);
	sh_n[tid] = min(sh_m[tid], bits);

	/* step 2: encode first n bits of bit plane */
	//y[tid] = stream[bidx].write_bits(y[tid], sh_m[tid]);
	y = write_bitters(bitter, make_bitter(y, 0), sh_m[tid], sbit);
	n = sh_n[tid];

	/* step 3: unary run-length encode remainder of bit plane */
	for (; n < size && bits && (bits-- && write_bitter(bitter, !!y, sbit)); y >>= 1, n++)
		for (; n < size - 1 && bits && (bits-- && !write_bitter(bitter, y & 1u, sbit)); y >>= 1, n++)
			;
	__syncthreads();


	sh_bitters[63 - tid] = bitter;
	sh_sbits[63 - tid] = sbit;

}
template<class Int, class UInt, class Scalar, uint bsize, int intprec>
__global__
void
__launch_bounds__(64,5)
cudaEncode
(
const unsigned long long count,
uint size,
const Scalar* data,
const unsigned char *g_cnt,
Word *blocks
)
{
  //	int mx = threadIdx.x + blockDim.x*blockIdx.x;
  //	int my = threadIdx.y + blockDim.y*blockIdx.y;
  //	int mz = threadIdx.z + blockDim.z*blockIdx.z;
  //	int eidx = mz*gridDim.x*blockDim.x*gridDim.y*blockDim.y + my*gridDim.x*blockDim.x + mx;

  extern __shared__ unsigned char smem[];


  uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
  uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
  uint bidx = idx*blockDim.x*blockDim.y*blockDim.z;

	unsigned char *sh_sbits;
	Bitter *sh_bitters;
	//int *sh_emax;
	uint *s_emax_bits;

	encode<Int, UInt, Scalar, bsize, intprec>(
		data,
		size, 

		smem, 

		sh_sbits,
		sh_bitters, 
		s_emax_bits,
		blocks
		);

  __syncthreads();
  if (tid == 0){
    uint tot_sbits = s_emax_bits[0];// sbits[0];
    uint  rem_sbits = s_emax_bits[0];// sbits[0];
    uint offset = 0;
    for (int i = 0; i < intprec && tot_sbits < c_maxbits; i++){
      if (sh_sbits[i] <= 64){
        write_outx<bsize>(sh_bitters, blocks + idx * bsize, rem_sbits, tot_sbits, offset, i, sh_sbits[i]);
      }
      else{
        write_outx<bsize>(sh_bitters, blocks + idx * bsize, rem_sbits, tot_sbits, offset, i, 64);
        write_outy<bsize>(sh_bitters, blocks + idx * bsize, rem_sbits, tot_sbits, offset, i, sh_sbits[i] - 64);
      }
    }
  }
}
template<class Int, class UInt, class Scalar, uint bsize, int intprec>
void encode
(
  int nx, int ny, int nz,
  const Scalar *d_data,
  thrust::device_vector<Word> &stream,
  const unsigned long long group_count,
  const uint size
)
{
  dim3 block_size, grid_size;
  thrust::device_vector<unsigned char> d_g_cnt;
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

	cudaEncode<Int, UInt, Scalar, bsize, intprec> << <grid_size, block_size, (2 * sizeof(unsigned char) + sizeof(Bitter) + sizeof(UInt) + sizeof(Int) + sizeof(Scalar) + 3 * sizeof(int)) * 64 + 32 * sizeof(Scalar) + 4 >> >
    (
    group_count, size,
    d_data,
    thrust::raw_pointer_cast(d_g_cnt.data()),
    thrust::raw_pointer_cast(stream.data())
    );
  cudaStreamSynchronize(0);
}

template<class Int, class UInt, class Scalar, uint bsize, int intprec>
void encode
(
int nx, int ny, int nz,
thrust::device_vector<Scalar> &d_data,
thrust::device_vector<Word > &stream,
const unsigned long long group_count,
const uint size
)
{
  encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz,
    thrust::raw_pointer_cast(d_data.data()),
    stream,
    group_count,
    size);
}

template<class Int, class UInt, class Scalar, uint bsize, int intprec>
void encode
(
int nx, int ny, int nz,
const thrust::host_vector<Scalar> &h_data,
thrust::device_vector<Word> &stream,
const unsigned long long group_count,
const uint size
)
{
  thrust::device_vector<unsigned char> d_g_cnt;

  thrust::device_vector<Scalar> d_data = h_data;

  encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, d_data, stream, group_count, size);
}
template<class Int, class UInt, class Scalar, uint bsize, int intprec>
void encode
(
int nx, int ny, int nz,
const thrust::host_vector<Scalar> &h_data,
thrust::host_vector<Word> &stream,
const unsigned long long group_count,
const uint size
)
{
  thrust::device_vector<Word > d_stream = stream;

  encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, h_data, d_stream, group_count, size);

  stream = d_stream;
}

}

#endif
