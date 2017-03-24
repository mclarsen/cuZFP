#include <assert.h>
#include "cuZFP.h"
#include "encode.cuh"
#include "ErrorCheck.h"
#include "decode.cuh"
#include <constant_setup.cuh>
#include <thrust/device_vector.h>
#include <iostream>

#define BSIZE  16
//uint MAXBITS = BSIZE*64;
//uint MAXPREC = 64;
//int MINEXP = -1074;
//const double rate = BSIZE;
//size_t  blksize = 0;
//uint size = 64;
//int EBITS = 11;                     /* number of exponent bits */
//const int EBIAS = 1023;
//const int intprec = 64;
namespace cuZFP {

void encode(int nx, int ny, int nz, std::vector<double> &in_data, EncodedData &encoded_data)
{
  ErrorCheck errors;
  assert(in_data.size() == nx * ny * nz);
  thrust::device_vector<double> d_in_data(in_data); 
  const size_t bits_per_val = 16;// THIS is bits per value 
                                // total allocated space is (num_values / num_blocks) * 64 bits per word /
  size_t total_blocks = in_data.size() / 64; 
  if(in_data.size() % 64 != 0) total_blocks++;
  const size_t bits_per_block = 64 * bits_per_val;
  const size_t bits_per_word = 64;
  const size_t total_bits = bits_per_block * total_blocks;
  const size_t alloc_size = total_bits / bits_per_word;
  thrust::device_vector<Word> d_encoded;
  d_encoded.resize(alloc_size);
  std::cout<<"Encoding\n";
  ConstantSetup::setup_3d(double() , BSIZE);
  encode<long long int, unsigned long long, double, 16, 64>(nx, ny, nz, d_in_data, d_encoded, 64); 
  errors.chk("Encode");
  encoded_data.m_data.resize(d_encoded.size());

  //thrust::copy(encoded_data.m_data.begin(), 
  //             encoded_data.m_data.end(),
  //             d_encoded.begin());
  Word * d_ptr = thrust::raw_pointer_cast(d_encoded.data());
  Word * h_ptr = &encoded_data.m_data[0];
  cudaMemcpy(h_ptr, d_ptr, alloc_size * sizeof(Word), cudaMemcpyDeviceToHost);
  encoded_data.m_dims[0] = nx;
  encoded_data.m_dims[1] = ny;
  encoded_data.m_dims[2] = nz;
}

void decode(const EncodedData &encoded_data, std::vector<double> &out_data)
{
  const int nx = encoded_data.m_dims[0];
  const int ny = encoded_data.m_dims[1];
  const int nz = encoded_data.m_dims[2];
  const size_t out_size = nx * ny * nz;

  thrust::device_vector<double> d_out_data(out_size); 
  thrust::device_vector<Word> d_encoded(encoded_data.m_data);

  decode<long long int, unsigned long long, double, 16, 64>(nx, ny, nz, d_encoded, d_out_data); 

  out_data.resize(out_size); 
  thrust::copy(d_out_data.begin(), 
               d_out_data.end(),
               out_data.begin());
}


} // namespace cuZFP

