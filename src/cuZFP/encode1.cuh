#ifndef CUZFP_ENCODE1_CUH
#define CUZFP_ENCODE1_CUH

//#include <helper_math.h>
#include "shared.h"
#include "ull128.h"
#include "BitStream.cuh"
#include "WriteBitter.cuh"
#include "shared.h"
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#include <cuZFP.h>
#include <debug_utils.cuh>
#include <type_info.cuh>

namespace cuZFP
{

template<typename Scalar, typename Int>
void 
inline __device__ floating_point_ops1(const int &tid,
                                      Int *sh_q,
                                      uint *s_emax_bits,
                                      const Scalar *sh_data,
                                      Scalar *sh_reduce,
                                      int *sh_emax,
                                      const Scalar &thread_val,
                                      Word *blocks,
                                      uint &blk_idx)
{

  /** FLOATING POINT ONLY ***/
  int max_exp = get_max_exponent1(tid, sh_data, sh_reduce);
	__syncthreads();

  /*** FLOATING POINT ONLY ***/
	Scalar w = quantize_factor(max_exp, Scalar());
  /*** FLOATING POINT ONLY ***/
  // block tranform
  sh_q[tid] = (Int)(thread_val * w); // sh_q  = signed integer representation of the floating point value
  /*** FLOATING POINT ONLY ***/
	if (tid == 0)
  {
		s_emax_bits[0] = 1;

		int maxprec = precision(max_exp, get_precision<Scalar>(), get_min_exp<Scalar>());

		uint e = maxprec ? max_exp + get_ebias<Scalar>() : 0;
		if(e)
    {
      // this is writing the exponent out
			blocks[blk_idx] = 2 * e + 1; // the bit count?? for this block
			s_emax_bits[0] = get_ebits<Scalar>() + 1;// this c_ebit = ebias
		}
	}
}


template<>
void 
inline __device__ floating_point_ops1<int,int>(const int &tid,
                                               int *sh_q,
                                               uint *s_emax_bits,
                                               const int *sh_data,
                                               int *sh_reduce,
                                               int *sh_emax,
                                               const int &thread_val,
                                               Word *blocks,
                                               uint &blk_idx)
{
  s_emax_bits[0] = 0;
  sh_q[tid] = thread_val;
}

template<>
void 
inline __device__ floating_point_ops1<long long int, long long int>(const int &tid,
                                     long long int *sh_q,
                                     uint *s_emax_bits,
                                     const long long int*sh_data,
                                     long long int *sh_reduce,
                                     int *sh_emax,
                                     const long long int &thread_val,
                                     Word *blocks,
                                     uint &blk_idx)
{
  s_emax_bits[0] = 0;
  sh_q[tid] = thread_val;
}

template<typename Scalar>
int
inline __device__
get_max_exponent1(const int &tid, 
                  const Scalar *sh_data,
                  Scalar *sh_reduce)
{
  Scalar val = sh_data[tid];
  const int offset = tid / 4 /*vals_per_block*/;
  const int local_pos = tid % 4;
	if (local_pos < 2)
  {
		sh_reduce[offset+local_pos] = 
      max(fabs(sh_data[offset+local_pos]), fabs(sh_data[offset+local_pos + 2]));
  }

	if (local_pos == 0)
  {
		sh_reduce[offset] = max(sh_reduce[offset], sh_reduce[offset + 1]);
	}

  printf("max exp %d of %f in offset %d\n", exponent(sh_reduce[offset]), val, offset);
	return exponent(sh_reduce[offset]);
}

//
//  Encode 1D array
//
template<typename Scalar>
__device__
void 
encode1(Scalar *sh_data,
	      const uint bsize, 
        unsigned char *smem,
        uint blk_idx,
        Word *blocks)
{
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;
  const int intprec = get_precision<Scalar>();

  // number of bits in the incoming type
  const uint size = sizeof(Scalar) * 8; 
  const uint vals_per_block = 4;
  const uint vals_per_cuda_block = 128;
  //shared mem that depends on scalar size
	__shared__ Scalar *sh_reduce;
	__shared__ Int *sh_q;
	__shared__ UInt *sh_p;

  // shared mem that always has the same size
	__shared__ int *sh_emax;
	__shared__ uint *sh_m, *sh_n;
	__shared__ unsigned char *sh_sbits;
	__shared__ Bitter *sh_bitters;
	__shared__ uint *s_emax_bits;

  //
  // These memory locations do not overlap (in time)
  // so we will re-use the same buffer to
  // conserve precious shared mem space
  //
	sh_reduce = &sh_data[0];
	sh_q = (Int*)&sh_data[0];
	sh_p = (UInt*)&sh_data[0];

	sh_sbits = &smem[0];
	sh_bitters = (Bitter*)&sh_sbits[vals_per_cuda_block];
	sh_m = (uint*)&sh_bitters[vals_per_cuda_block];
	sh_n = (uint*)&sh_m[vals_per_cuda_block];
	s_emax_bits = (uint*)&sh_n[vals_per_cuda_block];
  //TODO: this is different for 1D
	sh_emax = (int*)&s_emax_bits[1];

	uint tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z *blockDim.x*blockDim.y;

	Bitter bitter = make_bitter(0, 0);
	unsigned char sbit = 0;
	//uint kmin = 0;
  const uint word_bits = sizeof(Word) * 8;
  const uint total_words = (bsize * vals_per_cuda_block) / word_bits; 
  const uint words_per_block =  word_bits  / (bsize * vals_per_block);

  if(tid == 0)
  {
    printf("words per block %d \n", (int) words_per_block);
    printf("tot words per cuda block %d \n", (int) total_words);
  }
  // init output stream 
	if (tid < total_words)
		blocks[blk_idx + tid] = 0; 

  Scalar thread_val = sh_data[tid];
	__syncthreads();
  printf("val %f\n", thread_val); 

  //
  // this is basically a no-op for int types
  //
  floating_point_ops1(tid,
                      sh_q,
                      s_emax_bits,
                      sh_data,
                      sh_reduce,
                      sh_emax,
                      thread_val,
                      blocks,
                      blk_idx);
 
	__syncthreads();
  return;

}


template<class Scalar>
__global__
void __launch_bounds__(128,5)
cudaEncode1(const uint  bsize,
            const Scalar* data,
            Word *blocks,
            const int dim)
{
  extern __shared__ unsigned char smem[];
	__shared__ Scalar *sh_data;
	unsigned char *new_smem;

	sh_data = (Scalar*)&smem[0];
  //share data over 32 blocks( 4 vals * 32 blocks= 128) 
	new_smem = (unsigned char*)&sh_data[128];

  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  //
  //  The number of threads launched can be larger than total size of
  //  the array in cases where it cannot be devided into perfect block
  //  sizes. To account for this, we will clamp the values in each block
  //  to the bounds of the data set. 
  //

  const uint id = min(idx, dim - 1);
      
  const uint tid = threadIdx.x;
	sh_data[tid] = data[id];
  printf("tid %d data  %f\n",tid, sh_data[tid]);
	__syncthreads();
  const uint zfp_block_idx = idx % 4; 
  //if(tid == 0) printf("\n");
	encode1<Scalar>(sh_data,
                  bsize, 
                  new_smem,
                  zfp_block_idx,
                  blocks);

  __syncthreads();

}

void allocate_device_mem1d(const int encoded_dim, 
                           const int bsize, 
                           thrust::device_vector<Word> &stream)
{
  const size_t vals_per_block = 4;
  const size_t size = encoded_dim; 
  size_t total_blocks = size / vals_per_block; 
  const size_t bits_per_block = vals_per_block * bsize;
  const size_t bits_per_word = sizeof(Word) * 8;
  const size_t total_bits = bits_per_block * total_blocks;
  const size_t alloc_size = total_bits / bits_per_word;
  std::cout<<"Alloc 1D size "<<alloc_size<<"\n";
  stream.resize(alloc_size);
}

//
// Launch the encode kernel
//
template<class Scalar>
void encode1launch(int dim, 
                   const Scalar *d_data,
                   thrust::device_vector<Word> &stream,
                   const int bsize)
{
  std::cout<<"boomm\n";
  dim3 block_size, grid_size;
  const int cuda_block_size = 128;
  block_size = dim3(cuda_block_size, 1, 1);
  grid_size = dim3(dim, 1, 1);

  grid_size.x /= block_size.x; 

  // Check to see if we need to increase the block sizes
  // in the case where dim[x] is not a multiple of 4

  int encoded_dim = dim;

  if(dim % cuda_block_size != 0) 
  {
    grid_size.x++;
    encoded_dim = grid_size.x * cuda_block_size;
  }

  std::cout<<"allocating mem\n";
  allocate_device_mem1d(encoded_dim, bsize, stream);

  std::size_t shared_mem_size = sizeof(Scalar) * cuda_block_size 
                              + sizeof(Bitter) * cuda_block_size 
                              + sizeof(unsigned char) * cuda_block_size 
                              + sizeof(unsigned int) * 128 
                              + 2 * sizeof(int);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

  std::cout<<"event\n";
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout<<"Running kernel \n";
  std::cout<<"grid "<<grid_size.x<<" "<<grid_size.y<<" "<<grid_size.z<<"\n";
  std::cout<<"block "<<block_size.x<<" "<<block_size.y<<" "<<block_size.z<<"\n";
  cudaEventRecord(start);
	cudaEncode1<Scalar> << <grid_size, block_size, shared_mem_size>> >
    (bsize,
     d_data,
     thrust::raw_pointer_cast(stream.data()),
     dim);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  float rate = (float(dim) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Encode rate: %.2f (MB / sec)\n", rate);
}

//
// Encode a host vector and output a encoded device vector
//
template<class Scalar>
void encode1(int dim,
             thrust::device_vector<Scalar> &d_data,
             thrust::device_vector<Word> &stream,
             const int bsize)
{
  std::cout<<"inside encode\n";
  encode1launch<Scalar>(dim, thrust::raw_pointer_cast(d_data.data()), stream, bsize);
}

template<class Scalar>
void encode1(int dim,
             thrust::host_vector<Scalar> &h_data,
             thrust::host_vector<Word> &stream,
             const int bsize)
{
  thrust::device_vector<Word > d_stream = stream;
  thrust::device_vector<Scalar> d_data = stream;
  encode<Scalar>(dim, d_data, d_stream, bsize);
  stream = d_stream;
}

}

#endif
