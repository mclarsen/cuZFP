#ifndef CUZFP_H
#define CUZFP_H
#include <thrust/device_vector.h>
#include "encode.cuh"
#include "decode.cuh"

namespace cuZFP{

	template<class Int, class UInt, class Scalar, uint bsize, int intprec, typename BinaryFunction>
	__global__
		void
		//__launch_bounds__(64, 5)
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
		__shared__ Scalar *s_rhs, *s_lhs, *s_tmp;

		s_rhs = (Scalar*)&smem[0];
		s_tmp = (Scalar*)&s_rhs[64];
		s_lhs = (Scalar*)&s_tmp[64];

		unsigned char *new_smem = (unsigned char*)&s_lhs[64];

		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(lhs + idx*bsize, new_smem, tid, s_lhs);
		__syncthreads();
		cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(rhs + idx*bsize, new_smem, tid, s_rhs);
		__syncthreads();
		uint tidx = threadIdx.x + blockDim.x * blockIdx.x + (threadIdx.y + blockDim.y*blockIdx.y) * gridDim.x * blockDim.x + (threadIdx.z + blockDim.z * blockIdx.z)* gridDim.x * blockDim.x * gridDim.y * blockDim.y;
		//if (tidx == 99356){
		//	for (int i = 0; i < 4; i++){
		//		for (int j = 0; j < 4; j++){
		//			for (int k = 0; k < 4; k++){
		//				printf("h_u[%d] = %.30f;\n h_du[%d] = %.30f;\n", k + j * 64 + i * 4096, s_lhs[i * 16 + j * 4 + k], k + j * 64 + i * 4096, s_rhs[i * 16 + j * 4 + k]);

		//			}
		//		}
		//	}
		//	printf("wtf %d %f %f\n", tidx, s_lhs[tid], s_rhs[tid]);
		//}

		s_lhs[tid] = op(s_lhs[tid], s_rhs[tid]);
		__syncthreads();

		//if (tidx == 99356){
		//		for (int i = 0; i < 64; i++){
		//			printf("<%.30f %.30f >", s_lhs[i], s_rhs[i]);
		//		}
		//	printf("wtf %d %.30f %.30f\n", tidx, s_lhs[tid], s_rhs[tid]);
		//}
		//if (tid < bsize){
		//	lhs[idx * bsize + tid] = 0;
		//}

		cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(
			s_lhs,
			size,

			new_smem,

			idx * bsize,
			lhs
			);
		__syncthreads();
		//cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(lhs + idx*bsize, new_smem, tid, s_tmp);
		//__syncthreads();
		//if (s_tmp[0] > 1){
		//	//for (int i = 0; i < 64; i++){
		//	//	printf("<%f.30f %.30f %.30f >", s_tmp[i], s_lhs[i], s_rhs[i]);
		//	//}
		//	printf("wtf %d %.16f %.16f %.16f\n", tidx, s_tmp[tid], s_lhs[tid], s_rhs[tid]);

		//	s_lhs[tid] = 0;
		//	__syncthreads();
		//	cuZFP::encode<Int, UInt, Scalar, bsize, intprec>(
		//		s_lhs,
		//		size,

		//		new_smem,

		//		idx * bsize,
		//		lhs
		//		);
		//	__syncthreads();
		//	cuZFP::decode<Int, UInt, Scalar, bsize, intprec>(lhs + idx*bsize, new_smem, tid, s_tmp);
		//	printf("wtf %d %.16f %.16f %.16f\n", tidx, s_tmp[tid], s_lhs[tid], s_rhs[tid]);

		//}

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

			//cuZFP::decode needs  64 * (8*2) + 4 + 4 
			//of shmem
			//cuZFP::encode need (2 * sizeof(unsigned char) + sizeof(Bitter) + sizeof(UInt) + sizeof(Int) + sizeof(Scalar) + 3 * sizeof(int)) * 64 + 32 * sizeof(Scalar) + 4 
			//of shmem
			//obviously, take the larger of the two if you're doing both encode and decode

			transform<Int, UInt, Scalar, bsize, intprec > << <grid_size, block_size,
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
