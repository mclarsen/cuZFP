#include <iostream>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <algorithm>
#include <omp.h>
#include <fstream>

#define KEPLER 0
#include "ErrorCheck.h"
#include "include/encode.cuh"
#include "include/decode.cuh"
#include "include/cuZFP.cuh"

#include "zfparray3.h"

enum ENGHS_t{ N_LEFT, N_RIGHT, N_UP, N_DOWN, N_NEAR, N_FAR } enghs;

using namespace thrust;
using namespace std;

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const size_t nx = 512;
const size_t ny = 512;
const size_t nz = 512;
const int nt = 0;
const double pi = 3.14159265358979323846;


//BSIZE is the length of the array in class Bit
//It's tied to MAXBITS such that 
//MAXBITS = sizeof(Word) * BSIZE
//which is really
//MAXBITS = wsize * BSIZE
//e.g. if we match bits one-to-one, double -> unsigned long long
// then BSIZE = 64 and MAXPBITS = 4096
#define BSIZE  16
uint minbits = BSIZE*64;
uint MAXBITS = BSIZE*64;
uint MAXPREC = 64;
int MINEXP = -1074;
const double rate = BSIZE;
size_t  blksize = 0;
unsigned long long group_count = 0x46acca631ull;
uint size = 64;
int EBITS = 11;                     /* number of exponent bits */
const int EBIAS = 1023;
const int intprec = 64;


static const unsigned char
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


static size_t block_size(double rate) { return (lrint(64 * rate) + CHAR_BIT - 1) / CHAR_BIT; }


template<class Scalar>
void setupConst(const unsigned char *perm,
	uint maxbits_,
	uint maxprec_,
	int minexp_,
	int ebits_,
	int ebias_
	)
{
	ErrorCheck ec;
	ec.chk("setupConst start");
	cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char) * 64, 0); ec.chk("setupConst: c_perm");

	cudaMemcpyToSymbol(c_maxbits, &MAXBITS, sizeof(uint)); ec.chk("setupConst: c_maxbits");
	const uint sizeof_scalar = sizeof(Scalar);
	cudaMemcpyToSymbol(c_sizeof_scalar, &sizeof_scalar, sizeof(uint)); ec.chk("setupConst: c_sizeof_scalar");

	cudaMemcpyToSymbol(c_maxprec, &maxprec_, sizeof(uint)); ec.chk("setupConst: c_maxprec");
	cudaMemcpyToSymbol(c_minexp, &minexp_, sizeof(int)); ec.chk("setupConst: c_minexp");
	cudaMemcpyToSymbol(c_ebits, &ebits_, sizeof(int)); ec.chk("setupConst: c_ebits");
	cudaMemcpyToSymbol(c_ebias, &ebias_, sizeof(int)); ec.chk("setupConst: c_ebias");

	ec.chk("setupConst finished");



}



//Used to generate rand array in CUDA with Thrust
struct RandGen
{
	RandGen() {}

	__device__ float operator () (const uint idx)
	{
		thrust::default_random_engine randEng;
		thrust::uniform_real_distribution<float> uniDist(0.0, 0.0001);
		randEng.discard(idx);
		return uniDist(randEng);
	}
};



__device__
static inline
int idx(int x, int y, int z)
{
	return x +  y * (blockDim.x * gridDim.x) + z * (blockDim.x * gridDim.x * blockDim.y * gridDim.y);
}

template<typename Scalar>
__global__
void cudaDiffusion
(

	const Scalar *u,
	const Scalar dx,
	const Scalar dy,
	const Scalar dz,
	const Scalar dt,
	const Scalar k,
	const Scalar tfinal,

	Scalar *du
	
)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	Scalar uxx = (u[idx(max(0, x - 1), y, z)] - 2 * u[idx(x, y, z)] + u[idx(min(blockDim.x*gridDim.x - 1, x + 1), y, z)]) / (dx * dx);
	Scalar uyy = (u[idx(x, max(0, y - 1), z)] - 2 * u[idx(x, y, z)] + u[idx(x, min(blockDim.y*gridDim.y - 1, y + 1), z)]) / (dy * dy);
	Scalar uzz = (u[idx(x, y, max(0, z - 1))] - 2 * u[idx(x, y, z)] + u[idx(x, y, min(blockDim.z*gridDim.z-1, z + 1))]) / (dz * dz);

	du[idx(x, y, z)] = dt * k * (uxx + uyy + uzz);
}

template<typename Scalar>
__global__
void cudaSum
(

Scalar *u,
const Scalar *du

)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;
	u[idx(x, y, z)] += du[idx(x, y, z)];
}


template<class Int, class UInt, class Scalar, uint bsize, int intprec>
__global__
void
__launch_bounds__(64, 5)
cudaZFPDiffusion
(
const Scalar *u,
Word *du,
uint size,

const Scalar dx,
const Scalar dy,
const Scalar dz,
const Scalar dt,
const Scalar k
)
{
	uint x = threadIdx.x;
	uint y = threadIdx.y;
	uint z = threadIdx.z;

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint bidx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bdim = blockDim.x*blockDim.y*blockDim.z;
	uint tbidx = bidx*bdim;

	extern __shared__ unsigned char smem[];
	__shared__ Scalar *s_u, *s_du, *s_nghs, *s_u_ext;


	s_u = (Scalar*)&smem[0];
	s_du = (Scalar*)&s_u[64];
	s_u_ext = (Scalar*)&s_du[64];
	s_nghs = (Scalar*)&s_u_ext[216];

	unsigned char *new_smem = (unsigned char*)&s_nghs[64];

	//cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(u + bidx*bsize, new_smem, tid, s_u);
	//cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(du + bidx*bsize, new_smem, tid, s_du);
	//__syncthreads();

	int3 utid = make_int3(threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * blockIdx.y, threadIdx.z + blockDim.z * blockIdx.z);

	Scalar uxx = (u[idx(max(0, utid.x - 1), utid.y, utid.z)] - 2 * u[idx(utid.x, utid.y, utid.z)] + u[idx(min(blockDim.x*gridDim.x - 1, utid.x + 1), utid.y, utid.z)]) / (dx * dx);
	Scalar uyy = (u[idx(utid.x, max(0, utid.y - 1), utid.z)] - 2 * u[idx(utid.x, utid.y, utid.z)] + u[idx(utid.x, min(blockDim.y*gridDim.y - 1, utid.y + 1), utid.z)]) / (dy * dy);
	Scalar uzz = (u[idx(utid.x, utid.y, max(0, utid.z - 1))] - 2 * u[idx(utid.x, utid.y, utid.z)] + u[idx(utid.x, utid.y, min(blockDim.z*gridDim.z - 1, utid.z + 1))]) / (dz * dz);

	s_du[tid] = dt*k * (uxx + uyy + uzz);

	__syncthreads();

	cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(
		s_du,
		size,

		new_smem,

		bidx * bsize,
		du
		);
}
template<class Int, class UInt, class Scalar, uint bsize, int intprec>
__global__
void
__launch_bounds__(64, 5)
cudaZFPDiffusion
(
const Word *u,
Word *du,
uint size,

const Scalar dx,
const Scalar dy,
const Scalar dz,
const Scalar dt,
const Scalar k
)
{
	uint x = threadIdx.x;
	uint y = threadIdx.y;
	uint z = threadIdx.z;

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
	uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
	uint bdim = blockDim.x*blockDim.y*blockDim.z;
	uint bidx = idx*bdim;

	extern __shared__ unsigned char smem[];
	__shared__ Scalar *s_u, *s_du, *s_nghs, *s_u_ext;


	s_u = (Scalar*)&smem[0];
	s_du = (Scalar*)&s_u[64];
	s_u_ext = (Scalar*)&s_du[64];
	s_nghs = (Scalar*)&s_u_ext[216];

	unsigned char *new_smem = (unsigned char*)&s_nghs[64];

	cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(u + idx*bsize, new_smem, tid, s_u);
	//cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(du + idx*bsize, new_smem, tid, s_du);

	for (int i = 0; i < 3; i++){
		s_u_ext[i * 64 + tid] = 0;
	}

	if (tid < 24)
		s_u_ext[192 + tid] = 0;

	__syncthreads();

	//left
	s_nghs[tid] = 0;
	if (blockIdx.x > 0){
		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(u + ((blockIdx.x-1) + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*bsize, new_smem, tid, s_nghs);
	}
	__syncthreads();
	//if (tid == 0){
	//	for (int i = 0; i < 4; i++){
	//		for (int j = 0; j < 4; j++){
	//			s_u_ext[(i+1) * 6 + (j+1) * 36] = s_nghs[3 + i * blockDim.x + j * blockDim.x * blockDim.y];
	//		}
	//	}
	//}
	if (z == 0){
		s_u_ext[(x + 1) * 6 + (y + 1) * 36] = s_nghs[3 + x * blockDim.x + y * blockDim.x * blockDim.y];
	}

	__syncthreads();

	//right
	s_nghs[tid] = 0;
	if (blockIdx.x+1 < gridDim.x){
		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(u + (1 + blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*bsize, new_smem, tid, s_nghs);
	}
	__syncthreads();
	//if (tid == 0){
	//	for (int i = 0; i < 4; i++){
	//		for (int j = 0; j < 4; j++){
	//			s_u_ext[5 + (i+1) * 6 + (j+1) * 36] = s_nghs[i*blockDim.x + j * blockDim.x * blockDim.y];
	//		}
	//	}
	//}
	if (z == 0){
		s_u_ext[5 + (x + 1) * 6 + (y + 1) * 36] = s_nghs[x*blockDim.x + y * blockDim.x * blockDim.y];
	}
	__syncthreads();

	//down
	s_nghs[tid] = 0;
	if (blockIdx.y > 0){
		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(u + (blockIdx.x + (blockIdx.y - 1) * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*bsize, new_smem, tid, s_nghs);
	}
	__syncthreads();
	//if (tid == 0){
	//	for (int i = 0; i < 4; i++){
	//		for (int j = 0; j < 4; j++){
	//			s_u_ext[1 + i + (j+1) * 36] = s_nghs[i + 3*blockDim.x + j * blockDim.x * blockDim.y];
	//		}
	//	}
	//}
	if (z == 0){
		s_u_ext[1 + x + (y + 1) * 36] = s_nghs[x + 3 * blockDim.x + y * blockDim.x * blockDim.y];
	}
	__syncthreads();

	//up
	s_nghs[tid] = 0;
	if (blockIdx.y + 1 < gridDim.y){
		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(u + (blockIdx.x + (blockIdx.y + 1) * gridDim.x + blockIdx.z * gridDim.y * gridDim.x)*bsize, new_smem, tid, s_nghs);
	}
	__syncthreads();
	//if (tid == 0){
	//	for (int i = 0; i < 4; i++){
	//		for (int j = 0; j < 4; j++){
	//			s_u_ext[1 + i + 5*6 + (j+1) * 36] = s_nghs[i + j * blockDim.x * blockDim.y];
	//		}
	//	}
	//}
	if (z == 0){
		s_u_ext[1 + x + 5 * 6 + (y + 1) * 36] = s_nghs[x + y * blockDim.x * blockDim.y];
	}
	__syncthreads();

	//near
	s_nghs[tid] = 0;
	if (blockIdx.z > 0){
		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(u + (blockIdx.x + blockIdx.y * gridDim.x + (blockIdx.z - 1) * gridDim.y * gridDim.x)*bsize, new_smem, tid, s_nghs);
	}
	__syncthreads();
	//if (tid == 0){
	//	for (int i = 0; i < 4; i++){
	//		for (int j = 0; j < 4; j++){
	//			s_u_ext[1 + i + (j + 1) * 6] = s_nghs[i + (j)*blockDim.x + 3 * blockDim.x * blockDim.y];
	//		}
	//	}
	//}
	if (z == 0){
		s_u_ext[1 + x + (y + 1) * 6] = s_nghs[x + (y)*blockDim.x + 3 * blockDim.x * blockDim.y];
	}
	__syncthreads();

	//far
	s_nghs[tid] = 0;
	if (blockIdx.z + 1 < gridDim.z){
		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(u + (blockIdx.x + blockIdx.y * gridDim.x + (blockIdx.z + 1) * gridDim.y * gridDim.x)*bsize, new_smem, tid, s_nghs);
	}
	__syncthreads();
	//if (tid == 0){
	//	for (int i = 0; i < 4; i++){
	//		for (int j = 0; j < 4; j++){
	//			s_u_ext[1 + i + (j + 1) * 6 + 5 * 36] = s_nghs[i + (j)*blockDim.x ];
	//		}
	//	}
	//}
	if (z == 0){
		s_u_ext[1 + x + (y + 1) * 6 + 5 * 36] = s_nghs[x + (y)*blockDim.x];

	}
	__syncthreads();


	s_u_ext[1 + x + (y + 1) * 6 + (z + 1) * 36] = s_u[tid];
	__syncthreads();

	Scalar uxx = (s_u_ext[x + (y + 1) * 6 + (z + 1) * 36] - 2 * s_u_ext[x + 1 + (y + 1) * 6 + (z + 1) * 36] + s_u_ext[x + 2 + (y + 1) * 6 + (z + 1) * 36]) / (dx * dx);
	Scalar uyy = (s_u_ext[x + 1 + (y)* 6 + (z + 1) * 36] - 2 * s_u_ext[x + 1 + (y + 1) * 6 + (z + 1) * 36] + s_u_ext[x + 1 + (y + 2) * 6 + (z + 1) * 36]) / (dy * dy);
	Scalar uzz = (s_u_ext[x + 1 + (y + 1) * 6 + (z)* 36] - 2 * s_u_ext[x + 1 + (y + 1) * 6 + (z + 1) * 36] + s_u_ext[x + 1 + (y + 1) * 6 + (z + 2) * 36]) / (dz * dz);

	s_du[tid] = dt*k * (uxx + uyy + uzz);

	__syncthreads();
	//if (uxx < 0 || uyy < 0 || uzz < 0){
	//	printf("%d, %f, %f, %f, %f %f %f %d %d %d %d\n", tid, dt, k, s_du[tid], uxx, uyy, uzz, threadIdx.x + blockIdx.x * blockDim.x, threadIdx.y + blockIdx.y * blockDim.y, threadIdx.z + blockIdx.z * blockDim.z, threadIdx.x + blockIdx.x * blockDim.x + (threadIdx.y + blockIdx.y * blockDim.y)*gridDim.x * blockDim.x + (threadIdx.z + blockIdx.z * blockDim.z)*gridDim.x * blockDim.x * gridDim.y * blockDim.y);
	//}

	cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(
		s_du,
		size,

		new_smem,

		idx * bsize,
		du
		);
	//out[(threadIdx.z + blockIdx.z * 4)*gridDim.x * gridDim.y * blockDim.x * blockDim.y + (threadIdx.y + blockIdx.y * 4)*gridDim.x * blockDim.x + (threadIdx.x + blockIdx.x * 4)] = s_dblock[tid];
}



template<class Int, class UInt, class Scalar, uint bsize, int intprec>
void gpuZFPDiffusion
(
int nx, int ny, int nz,
device_vector<Word > &u,
device_vector<Word > &du,
device_vector<Scalar> &df_u,
const Scalar dx,
const Scalar dy,
const Scalar dz,
const Scalar dt,
const Scalar k,
const Scalar tfinal

)
{
	dim3 block_size = dim3(4, 4, 4);
	dim3 grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;

	cudaZFPDiffusion<Int, UInt, Scalar, bsize, intprec> << < grid_size, block_size, (sizeof(Scalar) * 2 + 2 * sizeof(unsigned char) + sizeof(Bitter) + sizeof(UInt) + sizeof(Int) + sizeof(Scalar) + 3 * sizeof(int)) * 64 + 32 * sizeof(Scalar) + 4 + 216 * sizeof(Scalar) >> >
		(
		thrust::raw_pointer_cast(u.data()),
		thrust::raw_pointer_cast(du.data()),
		size,
		dx,dy,dz,dt,k
		);
//	cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(
//		nx, ny, nz,
//		u, df_u,
//		group_count
//		);
//cudaZFPDiffusion<Int, UInt, Scalar, bsize, intprec> << < grid_size, block_size, (sizeof(Scalar) * 2 + 2 * sizeof(unsigned char) + sizeof(Bitter) + sizeof(UInt) + sizeof(Int) + sizeof(Scalar) + 3 * sizeof(int)) * 64 + 32 * sizeof(Scalar) + 4 + 216 * sizeof(Scalar) >> >
//	(
//	thrust::raw_pointer_cast(df_u.data()),
//	thrust::raw_pointer_cast(du.data()),
//	size,
//	dx,dy,dz,dt,k
//	);
	cuZFP::transform <Int, UInt, Scalar, bsize, intprec>
		(
		nx,ny,nz,
		size,
		u,
		du,
		thrust::plus<Scalar>()
		);

	//Scalar sum_u = cuZFP::reduce<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, u);
	//Scalar sum_du = cuZFP::reduce<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, du);

	//cout << "post-transform du: " << sum_du << " u: " << sum_u << endl;

}




template<class Int, class UInt, class Scalar, uint bsize>
void gpuEncode
(
host_vector<Scalar> &h_u
)
{
	
  device_vector<Scalar> d_u;
  d_u = h_u;



	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);



	dim3 emax_size(nx / 4, ny / 4, nz / 4);
	device_vector<Word > u(emax_size.x * emax_size.y * emax_size.z * bsize);

  cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, d_u, u, group_count, size);


	cudaStreamSynchronize(0);
	ec.chk("cudaEncode");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs/1000.0 << endl;
  cout << "sum: " << cuZFP::reduce<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, u) << endl;

	double tot_sum = 0, max_diff = 0, min_diff = 1e16;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

	cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, u, d_u, group_count);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&millisecs, start, stop);
  ec.chk("cudadecoe");

  cout << "decode GPU in time: " << millisecs / 1000.0 << endl;
  host_vector<Scalar> h_out = d_u;

	//array3d out(nx, ny, nz, rate);
	//for (int i = 0; i < h_out.size(); i++){
	//	out[i] = h_out[i];
	//}
}




int main()
{
	host_vector<double> h_vec_in(nx*ny*nz, 0);

  ifstream ifs("../../h512_0171_little_endian.raw", ios::binary);
  if (ifs) {
    double read;
    for (int i = 0; i < nx*ny*nz; i++){
      ifs.read(reinterpret_cast<char*>(&read), sizeof read);
      h_vec_in[i] = read;
    }
  }
  ifs.close();



  cout << "cpu encode start" << endl;
  double start_time = omp_get_wtime();
  zfp::array3d u(nx, ny, nz, rate);
  for (int i = 0; i < nx*ny*nz; i++){
    u[i] = h_vec_in[i];
  }
  double time = omp_get_wtime() - start_time;
  cout << "decode cpu time: " << time << endl;
  host_vector<double> h_vec_out(nx*ny*nz, 0);
  cout << "cpu decode start" << endl;
  start_time = omp_get_wtime();
  for (int z = 0; z < nz; z++){
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        h_vec_out[z*nx*ny + y*nx + x] = u(x, y, z);
      }
    }
  }
  time = omp_get_wtime() - start_time;
  cout << "decode cpu time: " << time << endl;
  cout << "sum: " << thrust::reduce(h_vec_out.begin(), h_vec_out.end()) << endl;

  cout << "GPU ZFP encode start" << endl;
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	setupConst<double>(perm, MAXBITS, MAXPREC, MINEXP, EBITS, EBIAS);
	cout << "Begin gpuDiffusion" << endl;
	gpuEncode<long long, unsigned long long, double, BSIZE>(h_vec_in);
	cout << "Finish gpuDiffusion" << endl;

}
