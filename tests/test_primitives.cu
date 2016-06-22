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

#define KEPLER 0
#include "ErrorCheck.h"
#include "include/encode.cuh"
#include "include/decode.cuh"
#include "include/cuZFP.cuh"

#include "array3d.h"
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

template<typename Scalar>
void gpu_discrete_sum
	(
	thrust::host_vector<Scalar> &h_val
	)

{

	thrust::device_vector<Scalar> d_val = h_val;


	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	Scalar sum = thrust::reduce(d_val.begin(), d_val.end());

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "GPU reduce in time: " << millisecs/1000.0 << " sum: " << sum << endl;

}
template<typename Scalar, typename BinaryFunction>
void gpu_discrete_transform
(
thrust::host_vector<Scalar> &h_lhs,
thrust::host_vector<Scalar> &h_rhs,
BinaryFunction op
)

{

	thrust::device_vector<Scalar> d_rhs = h_lhs;
	thrust::device_vector<Scalar> d_lhs = h_lhs;


	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	thrust::transform(d_lhs.begin(), d_lhs.end(), d_rhs.begin(), d_lhs.begin(), op);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	Scalar sum = thrust::reduce(d_lhs.begin(), d_lhs.end());

	cout << "GPU reduce in time: " << millisecs / 1000.0 << " sum: " << sum << endl;

}
template<class Int, class UInt, class Scalar, uint bsize>
void cuZFPReduce
(
host_vector<Scalar> &h_u
)
{

	ErrorCheck ec;
	dim3 emax_size(nx / 4, ny / 4, nz / 4);
	device_vector<Scalar> d_u;
	d_u = h_u;



	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);



	device_vector<Word > u(emax_size.x * emax_size.y * emax_size.z * bsize);
	cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, d_u, u, group_count, size);
	Scalar sum = cuZFP::reduce<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, u);



	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("reduce");
	cout << "GPU ZFP reduce in time: " << millisecs/1000.0 << " sum: " << sum << endl;
}

template<class Int, class UInt, class Scalar, uint bsize, typename BinaryFunction>
void cuZFPTransform
(
host_vector<Scalar> &h_lhs,
host_vector<Scalar> &h_rhs,
BinaryFunction op
)
{

	ErrorCheck ec;
	dim3 emax_size(nx / 4, ny / 4, nz / 4);
	device_vector<Scalar> d_lhs, d_rhs;
	d_lhs = h_lhs;
	d_rhs = h_rhs;


	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);



	device_vector<Word > lhs(emax_size.x * emax_size.y * emax_size.z * bsize);
	device_vector<Word > rhs(emax_size.x * emax_size.y * emax_size.z * bsize);
	cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, d_lhs, lhs, group_count, size);
	cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, d_rhs, rhs, group_count, size);
	cuZFP::transform<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, size, lhs, rhs, op);


	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("reduce");
	Scalar sum = cuZFP::reduce<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, lhs);
	cout << "GPU ZFP reduce in time: " << millisecs / 1000.0 << " sum: " << sum << endl;
}
template<class Scalar, typename Array>
void discrete_sum
(
	host_vector<Scalar> &h_val
)
{
	double start_time = omp_get_wtime();
	Array u(nx, ny, nz, rate);
	for (int z = 0; z < nz; z++){
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				u[x + y * nx + z * nx * ny] = h_val[x + y * nx + z * nx * ny];

			}
		}
	}

	//for (int i = 0; i < nx*ny*nz; i++){
	//	u[i] = h_val[i];
	//}

	// compute du/dt
	double sum = 0.0;
	for (int z = 0; z < nz; z++){
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				sum += u(x, y, z);
			}
		}
	}

	double time = omp_get_wtime() - start_time;
	cout << "discrete time: " << time << " sum: " << sum << endl;
}

template<class Scalar, typename Array, typename BinaryFunction>
void discrete_transform
(
host_vector<Scalar> &h_lhs, 
host_vector<Scalar> &h_rhs,
BinaryFunction op

)
{
	double start_time = omp_get_wtime();
	Array lhs(nx, ny, nz, rate), rhs(nx, ny, nz, rate);
	for (int z = 0; z < nz; z++){
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				rhs[x + y * nx + z * nx * ny] = h_rhs[x + y * nx + z * nx * ny];
				lhs[x + y * nx + z * nx * ny] = h_lhs[x + y * nx + z * nx * ny];
			}
		}
	}

	// compute du/dt
	for (int z = 0; z < nz; z++){
		for (int y = 0; y < ny; y++) {
			for (int x = 0; x < nx; x++) {
				lhs[x + y * nx + z * nx * ny] = op(lhs[x + y * nx + z * nx * ny], rhs[x + y * nx + z * nx * ny]);
			}
		}
	}

	double time = omp_get_wtime() - start_time;

	double sum = 0.0;
	for (int i = 0; i < nx*ny*nz; i++){
		sum += lhs[i]; 
	}
	cout << "discrete time: " << time << " sum: " << sum << endl;
}


int main()
{
	host_vector<double> h_vec_in(nx*ny*nz, 0);

	device_vector<double> d_vec_in(nx*ny*nz);
		thrust::counting_iterator<uint> index_sequence_begin(0);
	thrust::transform(
		index_sequence_begin,
		index_sequence_begin + nx*ny*nz,
		d_vec_in.begin(),
		RandGen());

	h_vec_in = d_vec_in;
	d_vec_in.clear();
	d_vec_in.shrink_to_fit();



	cout << "cpu sum start" << endl;

	discrete_sum<double, array3d>(h_vec_in);
	cout << "compressed cpu sum start" << endl;
	discrete_sum<double, zfp::array3d>(h_vec_in);

	cout << "GPU discete reduce start" << endl;
	gpu_discrete_sum<double>(h_vec_in);

	cout << "GPU ZFP reduce start" << endl;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	setupConst<double>(perm, MAXBITS, MAXPREC, MINEXP, EBITS, EBIAS);
	cout << "Begin cuZFPReduce" << endl;
	cuZFPReduce<long long, unsigned long long, double, BSIZE>(h_vec_in);
	cout << "Finish cuZFPReduce" << endl;


	host_vector<double> h_rhs(nx*ny*nz, 0.0);
	device_vector<double> d_rhs(nx*ny*nz);

	thrust::transform(
		index_sequence_begin,
		index_sequence_begin + nx*ny*nz,
		d_rhs.begin(),
		RandGen());
	h_rhs = d_rhs;
	d_rhs.clear();
	d_rhs.shrink_to_fit();




	cout << "cpu transform start" << endl;
	discrete_transform<double, array3d>(h_vec_in, h_rhs, thrust::plus<double>());
	cout << "compressed cpu transform start" << endl;
	discrete_transform<double, zfp::array3d>(h_vec_in, h_rhs, thrust::plus<double>());

	cout << "GPU discete transfor start" << endl;
	gpu_discrete_transform<double>(h_vec_in, h_rhs, thrust::plus<double>());

	cout << "Begin cuZFPTransform" << endl;
	cuZFPTransform<long long, unsigned long long, double, BSIZE>(h_vec_in, h_rhs, thrust::plus<double>());
	cout << "Finish cuZFPTransform" << endl;
}
