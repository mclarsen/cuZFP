#ifndef CUZFP_ALIGNED_WRITER_CUH
#define CUZFP_ALIGNED_WRITER_CUH

#include "shared.h"

namespace cuZFP
{
struct aligned_writer
{
  const int m_bits_per_block;

  __device__ aligned_writer(const int &bits_per_block)
    : m_bits_per_block(bits_per_block) 
  {}

  
  void inline __device__ write_exponet(Word stream[], const int &blk_index, const int &exp)
  {
    const int offset = (blk_index * m_bits_per_block) % (sizeof(Word) * 8); 
    const int index = (blk_index * m_bits_per_block) / (sizeof(Word) * 8); 
    atomicAdd(&stream[index], exponent << offset); 
  }


}; // aligned writer

}// namespace cuZFP
#endif
