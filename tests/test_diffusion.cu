﻿#include <iostream>
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

#include "array3d.h"
#include "zfparray3.h"


using namespace thrust;
using namespace std;

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const size_t nx = 64;
const size_t ny = 64;
const size_t nz = 64;
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

template<typename Array>
void rme
(
const Array &u,
int x0,
int y0,
int z0,
const double dx,
const double dy,
const double dz,
const double k,

double t
)
{

	// compute root mean square error with respect to exact solution
	double e = 0;
	double sum = 0;
	for (int z = 1; z < nz - 1; z++){
		double pz = dz * (z - z0);
		for (int y = 1; y < ny - 1; y++) {
			double py = dy * (y - y0);
			for (int x = 1; x < nx - 1; x++) {
				double px = dx * (x - x0);
				double f = u(x, y, z);
				//http://nptel.ac.in/courses/105103026/34
				double g = dx * dy * dz * std::exp(-(px * px + py * py + pz * pz) / (4 * k * t)) / powf(4 * pi * k * t, 3.0 / 2.0);
				e += (f - g) * (f - g);
				sum += f;
			}
		}
	}

	e = std::sqrt(e / ((nx - 2) * (ny - 2)));
	std::cerr.unsetf(std::ios::fixed);
	std::cerr << "rate=" << rate << " sum=" << std::fixed << sum << " error=" << std::setprecision(6) << std::scientific << e << std::endl;

}

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

template<typename Scalar>
void gpu_discrete_solution
	(
	const int x0,
	const int y0,
	const int z0,
		const Scalar dx,
		const Scalar dy,
		const Scalar dz,
		const Scalar dt,
		const Scalar k,
		const Scalar tfinal
		)
{
	thrust::host_vector<Scalar> h_u(nx*ny*nz, 0);

	thrust::device_vector<Scalar> u(nx*ny*nz);
	thrust::device_vector<Scalar> du(nx*ny*nz);
	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	h_u[x0 + y0 * nx + z0 * nx * ny] = 1; 
	u = h_u;

	dim3 block_size(4, 4, 4);
	dim3 grid_size;
	grid_size.x = nx / block_size.x;
	grid_size.y = ny / block_size.y;
	grid_size.z = nz / block_size.z;

	double t;
	for (t = 0; t < tfinal; t += dt) {
		//std::cerr << "t=" << std::fixed << t << std::endl;
		cudaDiffusion << <grid_size, block_size >> >
			(
			thrust::raw_pointer_cast(u.data()),
			dx,dy,dz,
			dt,
			k,
			tfinal,
			thrust::raw_pointer_cast(du.data())
			);
		cudaStreamSynchronize(0);
		ec.chk("cudaDiffusion");

		cudaSum << < grid_size, block_size >> >
			(
			thrust::raw_pointer_cast(u.data()),
			thrust::raw_pointer_cast(du.data())
			);
		//thrust::transform(
		//	u.begin(), // begin input (1) iterator
		//	u.end(),   // end input (1) iterator
		//	du.begin(),  // begin input (2) iteratorout
		//	u.begin(),
		//	thrust::plus<Scalar>()
		//	);
	}

	h_u = u;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs << endl;

	array3d out(nx, ny, nz, 0);

	for (int i = 0; i < u.size(); i++){
		out[i] = h_u[i];
	}

	rme(out, x0, y0, z0, dx, dy, dz, k, tfinal - dt);
}
template<class Int, class UInt, class Scalar, uint bsize>
void gpuTestBitStream
(
host_vector<Scalar> &h_data
)
{
	host_vector<int> h_emax;
	host_vector<UInt> h_p;
	host_vector<Int> h_q;
	host_vector<UInt> h_buf(nx*ny*nz);
	host_vector<cuZFP::Bit<bsize> > h_bits;
	device_vector<unsigned char> d_g_cnt;

  device_vector<Scalar> data;
  data = h_data;


	dim3 emax_size(nx / 4, ny / 4, nz / 4);

	dim3 block_size(8, 8, 8);
	dim3 grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	//const uint kmin = intprec > maxprec ? intprec - maxprec : 0;

	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);



  device_vector<Word > block(emax_size.x * emax_size.y * emax_size.z * bsize);
  cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, data, block, group_count, size);

	cudaStreamSynchronize(0);
	ec.chk("cudaEncode");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs << endl;

  thrust::host_vector<Word > cpu_block;
  cpu_block = block;
	UInt sum = 0;
  for (int i = 0; i < cpu_block.size(); i++){
    sum += cpu_block[i];
	}
	cout << "encode UInt sum: " << sum << endl;

  cudaMemset(thrust::raw_pointer_cast(data.data()), 0, sizeof(Scalar)*data.size());

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);


	cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(nx, ny, nz, block, data, group_count);

  ec.chk("cudaDecode");
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);

	cout << "decode parallel GPU in time: " << millisecs << endl;

	double tot_sum = 0, max_diff = 0, min_diff = 1e16;

	host_vector<Scalar> h_out = data;
	for (int i = 0; i < h_data.size(); i++){
		int k = 0, j = 0;
		frexp(h_data[i], &j);
		frexp(h_out[i], &k);

		//if (abs(j - k) > 1){
		//	cout << i << " " << j << " " << k << " " << h_data[i] << " " << h_out[i] << endl;
		//	//exit(-1);
		//}
		double diff = fabs(h_data[i] - h_out[i]);
		//if (diff > 1 )
		//	cout << i << " " << j << " " << k << " " << h_data[i] << " " << h_out[i] << endl;

		if (max_diff < diff)
			max_diff = diff;
		if (min_diff > diff)
			min_diff = diff;

		tot_sum += diff;
	}

	cout << "tot diff: " << tot_sum << " average diff: " << tot_sum / (float)h_data.size() << " max diff: " << max_diff << " min diff: " << min_diff << endl;
	cout << "sum: " << thrust::reduce(h_data.begin(), h_data.end()) << " " << thrust::reduce(h_out.begin(), h_out.end()) << endl;

	//gpuValidate<Int, UInt, Scalar, bsize>(h_data, q, data);

}

template<typename Array>
void discrete_solution
(
Array &u,

int x0, int y0, int z0,
const double dx,
const double dy,
const double dz,
const double dt,
const double k,
const double tfinal
)
{
	// initialize u (constructor zero-initializes)
	//rate = u.rate();
	double start_time = omp_get_wtime();

	u(x0, y0, z0) = 1;

	// iterate until final time
	std::cerr.precision(6);
	double t;
	for (t = 0; t < tfinal; t += dt) {
		//std::cerr << "t=" << std::fixed << t << std::endl;
		// compute du/dt
		Array du(nx, ny, nz, rate);
		for (int z = 1; z < nz - 1; z++){
			for (int y = 1; y < ny - 1; y++) {
				for (int x = 1; x < nx - 1; x++) {
					double uxx = (u(x - 1, y, z) - 2 * u(x, y, z) + u(x + 1, y, z)) / (dx * dx);
					double uyy = (u(x, y - 1, z) - 2 * u(x, y, z) + u(x, y + 1, z)) / (dy * dy);
					double uzz = (u(x, y, z - 1) - 2 * u(x, y, z) + u(x, y, z + 1)) / (dz * dz);
					du(x, y, z) = dt * k * (uxx + uyy + uzz);
				}
			}
		}
		// take forward Euler step
		for (uint i = 0; i < u.size(); i++)
			u[i] += du[i];
	}
	double time = omp_get_wtime() - start_time;
	cout << "discrete time: " << time << endl;

	rme(u, x0, y0, z0, dx, dy, dz, k, tfinal - dt);

}


int main()
{
	host_vector<double> h_vec_in(nx*ny*nz);

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

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	setupConst<double>(perm, MAXBITS, MAXPREC, MINEXP, EBITS, EBIAS);
	cout << "Begin gpuTestBitStream" << endl;
  gpuTestBitStream<long long, unsigned long long, double, BSIZE>(h_vec_in);
	cout << "Finish gpuTestBitStream" << endl;



	// location of point heat source
	int x0 = (nx - 1) / 2;
	int y0 = (ny - 1) / 2;
	int z0 = (nz - 1) / 2;
	// constants used in the solution
	const double k = 0.04;
	const double dx = 2.0 / (std::max(nz,std::max(nx, ny)) - 1);
	const double dy = 2.0 / (std::max(nz, std::max(nx, ny)) - 1);
	const double dz = 2.0 / (std::max(nz, std::max(nx, ny)) - 1);
	const double dt = 0.5 * (dx * dx + dy * dy) / (8 * k);
	const double tfinal = nt ? nt * dt : 1;

	zfp::array3d u(nx, ny, nz, rate);

	discrete_solution<zfp::array3d>(u, x0, y0, z0, dx,dy,dz,dt,k, tfinal);

	array3d u2(nx, ny, nz, rate);
	discrete_solution<array3d>(u2, x0, y0, z0, dx, dy, dz, dt, k, tfinal);

	gpu_discrete_solution<double>(x0, y0, z0, dx, dy, dz, dt, k, tfinal);

}