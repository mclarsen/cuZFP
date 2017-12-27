#ifndef CUZFP_DECODE1_CUH
#define CUZFP_DECODE1_CUH

#include "shared.h"
#include <thrust/device_vector.h>
#include <type_info.cuh>

namespace cuZFP {

template<typename Scalar>
__device__ 
Scalar  decode1(const Word *blocks,
                unsigned char *smem,
                const uint bsize)
{
  typedef typename zfp_traits<Scalar>::UInt UInt;
  typedef typename zfp_traits<Scalar>::Int Int;
  const int intprec = get_precision<Scalar>();
  return 0;
}

template<int block_size>
struct BlockReader
{
  int m_current_bit;
  const int m_bsize; 
  Word *m_words;
  Word m_buffer;
  bool m_valid_block;
  __device__ BlockReader(Word *b, const int &bsize, const int &block_idx, const int &num_blocks)
    :  m_bsize(bsize), m_valid_block(true)
  {
    if(block_idx >= num_blocks) m_valid_block = false;
    int word_index = (block_idx * bsize * block_size)  / (sizeof(Word) * 8); 
    m_words = b + word_index;
    m_buffer = *m_words;
    m_current_bit = (block_idx * bsize * block_size) % (sizeof(Word) * 8); 
    m_buffer >>= m_current_bit;
    //if(threadIdx.x ==0 ) print_bits(m_buffer);
  }

  inline __device__ 
  uint read_bit()
  {
    uint bit = m_buffer & 1;
    ++m_current_bit;
    m_buffer >>= 1;
    // handle moving into next word
    if(m_current_bit >= sizeof(Word) * 8) 
    {
      m_current_bit = 0;
      ++m_words;
      m_buffer = *m_words;
    }
    return bit; 
  }


  // note this assumes that n_bits is <= 64
  inline __device__ 
  uint read_bits(const int &n_bits)
  {
    uint bits; 
    // rem bits will always be positive
    int rem_bits = sizeof(Word) * 8 - m_current_bit;
    
    int first_read = min(rem_bits, n_bits);

    // first mask 
    Word mask = ((Word)1<<((first_read)))-1;
    bits = m_buffer & mask;
    m_buffer >>= n_bits;

    int next_read = 0;
    if(n_bits >= rem_bits) 
    {
      //need to go into next word
      m_words++;
      m_buffer = *m_words;
      m_current_bit = 0;
      next_read = n_bits - first_read; 
    }
   
    // this is basically a no-op when first read constained 
    // all the bits. TODO: if we have aligned reads, this could 
    // be a conditional without divergence
    mask = ((Word)1<<((next_read)))-1;
    bits += (m_buffer & mask) << first_read;
    m_buffer >>= next_read;
    m_current_bit += next_read; 
    //printf(" outputing n = %d bits first read %d : ", n_bits, first_read);
    //print_bits(bits);  
    return bits;
  }

  private:
  __device__ BlockReader()
  {
  }

}; // block reader

template<typename Scalar, int Size, typename UInt>
inline __device__
void decode_ints(BlockReader<Size> &reader, uint &max_bits, UInt *data)
{
  const int intprec = get_precision<Scalar>();
  //memset(data, 0, sizeof(UInt) * Size);
  unsigned int x; 
  // maxprec = 64;
  const uint kmin = 0; //= intprec > maxprec ? intprec - maxprec : 0;
  int bits = max_bits;
  for (uint k = intprec, n = 0; bits && k-- > kmin;)
  {
    //printf("plane %d : ", k);
    //print_bits(reader.m_buffer);
    // read bit plane
    uint m = MIN(n, bits);
    bits -= m;
    x = reader.read_bits(m);
    for (; n < Size && bits && (bits--, reader.read_bit()); x += (Word) 1 << n++)
      for (; n < (Size - 1) && bits && (bits--, !reader.read_bit()); n++);
    
    //printf("x = %d\n", (int) x);
    // deposit bit plane
    #pragma unroll
    for (int i = 0; x; i++, x >>= 1)
    {
      data[i] += (UInt)(x & 1u) << k;
    }

  } 
}


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
    //for(int a = 0; a < 128; ++a) 
    //{
    //  __syncthreads();
    //  if(threadIdx.x == a)
    //  {
    //    for(int i = 0; i < 4; ++i)
    //    {
    //      printf("block %d data at %d = %d\n", block_idx, i, (int) data[i]);
    //    }
    //  }
    //}
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
    //    printf("data at %d = %d\n", i, (int) result[i]);
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
  int zfp_blocks = (dim + (4 - dim % 4)) / 4;
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
