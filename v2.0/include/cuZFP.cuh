#ifndef CUZFP_H
#define CUZFP_H
#include <thrust/device_vector.h>
#include "encode.cuh"
#include "decode.cuh"

namespace cuZFP{

	template<class Int, class UInt, class Scalar, uint bsize, int intprec, typename BinaryFunction>
	__global__
		void
		__launch_bounds__(64, 5)
		transform
		(
		Word *lhs,
		const Word *rhs,
		uint size,
		BinaryFunction op
		)
	{
		uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
		uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
		uint bdim = blockDim.x*blockDim.y*blockDim.z;
		uint bidx = idx*bdim;

		extern __shared__ unsigned char smem[];
		__shared__ Scalar *s_rhs, *s_lhs;

		s_rhs = (Scalar*)&smem[0];
		s_lhs = (Scalar*)&s_rhs[64];

		unsigned char *new_smem = (unsigned char*)&s_lhs[64];


		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(lhs + idx*bsize, new_smem, tid, s_lhs);
		__syncthreads();
		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(rhs + idx*bsize, new_smem, tid, s_rhs);
		__syncthreads();

		s_lhs[tid] = op(s_lhs[tid], s_rhs[tid]);
		__syncthreads();

		cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(
			s_lhs,
			size,

			new_smem,

			idx * bsize,
			lhs
			);
		//out[(threadIdx.z + blockIdx.z * 4)*gridDim.x * gridDim.y * blockDim.x * blockDim.y + (threadIdx.y + blockIdx.y * 4)*gridDim.x * blockDim.x + (threadIdx.x + blockIdx.x * 4)] = s_dblock[tid];
	}
	template<class Int, class UInt, class Scalar, uint bsize, int intprec, typename BinaryFunction>
		void
			transform
			(
			uint nx, uint ny, uint nz,
			uint size,
			thrust::device_vector<Word> &lhs,
			thrust::device_vector<Word> &rhs,
			BinaryFunction op
			)
		{
			const dim3 block_size(4, 4, 4);

			const dim3 grid_size(nx / block_size.x, ny / block_size.y, nz / block_size.z);

			transform<Int, UInt, Scalar, bsize, intprec> << <grid_size, block_size, 
			(sizeof(Scalar) * 2 + 2 * sizeof(unsigned char) + sizeof(Bitter) + sizeof(UInt) + sizeof(Int) + sizeof(Scalar) + 3 * sizeof(int)) * 64 + 32 * sizeof(Scalar) + 4 >> >
				(
				thrust::raw_pointer_cast(lhs.data()),
				thrust::raw_pointer_cast(rhs.data()),
				size,
				op
				);

		}


	template<class Int, class UInt, class Scalar, uint bsize, int intprec>
	__global__
		void
		__launch_bounds__(64, 5)
		cudaReduce
		(
		const Word *val,
		Scalar *out
		)
	{
		uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;
		uint idx = (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x);
		uint bdim = blockDim.x*blockDim.y*blockDim.z;
		uint bidx = idx*bdim;

		extern __shared__ unsigned char smem[];
		__shared__ Scalar *s_val, *sh_reduce;

		s_val = (Scalar*)&smem[0];
		sh_reduce = (Scalar*)&s_val[64];

		unsigned char *new_smem = (unsigned char*)&sh_reduce[64];


		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(val + idx*bsize, new_smem, tid, s_val);
		__syncthreads();

		if (tid < 32)
			sh_reduce[tid] =s_val[tid] + s_val[tid + 32];
		if (tid < 16)
			sh_reduce[tid] = sh_reduce[tid] + sh_reduce[tid + 16];
		if (tid < 8)
			sh_reduce[tid] = sh_reduce[tid] + sh_reduce[tid + 8];
		if (tid < 4)
			sh_reduce[tid] = sh_reduce[tid] + sh_reduce[tid + 4];
		if (tid < 2)
			sh_reduce[tid] = sh_reduce[tid] + sh_reduce[tid + 2];
		if (tid == 0){
			sh_reduce[0] = sh_reduce[tid] + sh_reduce[tid + 1];
		}

		out[idx] = sh_reduce[0]; 
	}

	template<class Int, class UInt, class Scalar, uint bsize, int intprec>
		Scalar
		reduce
		(
		uint nx, uint ny, uint nz,

			thrust::device_vector<Word> &val,
			Scalar init = 0
		)
	{
		const dim3 block_size(4, 4, 4);
		const dim3 grid_size(nx / block_size.x, nx / block_size.y, nx / block_size.z);

		thrust::device_vector<Scalar> tmp(grid_size.x*grid_size.y*grid_size.z);
		cudaReduce<Int, UInt, Scalar, bsize, intprec> << < grid_size, block_size, 64 * 8 * 2 + 64 * (8) + 4 + 4 >> >
			(
			thrust::raw_pointer_cast(val.data()),
			thrust::raw_pointer_cast(tmp.data())
			);

		return thrust::reduce(tmp.begin(), tmp.end(), (Scalar)init);
	}
		
};
#endif
