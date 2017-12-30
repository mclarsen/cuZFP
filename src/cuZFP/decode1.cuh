#ifndef CUZFP_DECODE1_CUH
#define CUZFP_DECODE1_CUH

#include "shared.h"
#include <thrust/device_vector.h>
#include <type_info.cuh>

namespace cuZFP {


template<class Scalar>
__global__
void
cudaDecode1(Word *blocks,
            Scalar *out,
            const int dim,
            uint bsize)
{
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;
  const int intprec = get_precision<Scalar>();

  const int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_blocks = (dim + (4 - dim % 4)) / 4;
  if(dim % 4 != 0) total_blocks = (dim + (4 - dim % 4)) / 4;
  if(block_idx >= total_blocks) return;
  BlockReader<4> reader(blocks, bsize, block_idx, total_blocks);
 
  Scalar result[4] = {0,0,0,0};

  uint s_cont = 1;
  //
  // there is no skip path for integers so just continue
  //
  if(!is_int<Scalar>())
  {
    s_cont = reader.read_bit();
  }

  if(s_cont)
  {
    uint ebits = get_ebits<Scalar>() + 1;

    uint emax;
    if(!is_int<Scalar>())
    {
      // read in the shared exponent
      emax = reader.read_bits(ebits - 1) - get_ebias<Scalar>();
    }
    else
    {
      // no exponent bits
      ebits = 0;
    }

    const uint vals_per_block = 4;
    uint maxbits = bsize * vals_per_block;
	  maxbits -= ebits;
    
    UInt data[vals_per_block];
    decode_ints<Scalar, 4, UInt>(reader, maxbits, data);
    Int iblock[4];
    #pragma unroll 4
    for(int i = 0; i < 4; ++i)
    {
      // cperm
		  iblock[i] = uint2int(data[i]);
    }

    inv_lift<Int,1>(iblock);

		Scalar inv_w = dequantize<Int, Scalar>(1, emax);
    
    //if(threadIdx.x == 0) printf("inv %d \n", inv_w);

    #pragma unroll 4
    for(int i = 0; i < 4; ++i)
    {
		  result[i] = inv_w * (Scalar)iblock[i];
    }
     
    //if(block_idx == 0)
    //{
    //  for(int i = 0; i < 4; ++i)
    //  {
    //    printf("data at %d = %f\n", i,  result[i]);
    //  }
    //}

  }

  // TODO dim could end in the middle of this block
  //printf("thread  = %d \n", block_idx);
  if(block_idx < total_blocks)
  {
    //if(threadIdx.x == 0) printf("inv %d \n", block_idx);

    const int offset = block_idx * 4;
    out[offset + 0] = result[0];
    out[offset + 1] = result[1];
    out[offset + 2] = result[2];
    out[offset + 3] = result[3];
    //if(threadIdx.x==0) printf("out data %d\n", out[offset+0]);
  }
  // write out data
}
template<class Scalar>
void decode1(int dim, 
             thrust::device_vector<Word> &stream,
             Scalar *d_data,
             uint bsize)
{
  const int block_size_dim = 128;
  int zfp_blocks = dim / 4;
  if(dim % 4 != 0)  zfp_blocks = (dim + (4 - dim % 4)) / 4;
  //int block_pad = block_size_dim - zfp_blocks; 
  int block_pad = 0;
  if(zfp_blocks % block_size_dim != 0) block_pad = block_size_dim - zfp_blocks % block_size_dim; 
  dim3 block_size = dim3(block_size_dim, 1, 1);
  dim3 grid_size = dim3(block_pad + zfp_blocks, 1, 1);

  grid_size.x /= block_size.x; 
  std::cout<<"Dims "<< dim<<" zfp blocks "<<zfp_blocks<<"\n";
  std::cout<<"Decode1 dims \n";
  std::cout<<"grid "<<grid_size.x<<" "<<grid_size.y<<" "<<grid_size.z<<"\n";
  std::cout<<"block "<<block_size.x<<" "<<block_size.y<<" "<<block_size.z<<"\n";
  // setup some timing code
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  cudaDecode1<Scalar> << < grid_size, block_size >> >
    (raw_pointer_cast(stream.data()),
		 d_data,
     dim,
     bsize);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
	cudaStreamSynchronize(0);

  float miliseconds = 0;
  cudaEventElapsedTime(&miliseconds, start, stop);
  float seconds = miliseconds / 1000.f;
  printf("Decode elapsed time: %.5f (s)\n", seconds);
  float rate = (float(dim) * sizeof(Scalar) ) / seconds;
  rate /= 1024.f;
  rate /= 1024.f;
  printf("Decode rate: %.2f (MB / sec)\n", rate);
}

template<class Scalar>
void decode1(int dim, 
             thrust::device_vector<Word > &block,
             thrust::device_vector<Scalar> &d_data,
             uint bsize)
{
	decode1<Scalar>(dim, block, thrust::raw_pointer_cast(d_data.data()), bsize);
}

} // namespace cuZFP

#endif
