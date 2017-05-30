#include <assert.h>
#include "cuZFP.h"
#include "encode.cuh"
#include "ErrorCheck.h"
#include "decode.cuh"
#include <constant_setup.cuh>
#include <thrust/device_vector.h>
#include <iostream>

#define BSIZE 8 
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
   
  int3 dims = make_int3(nx, ny, nz);
  const int bsize = encoded_data.m_bsize;

  assert(BSIZE == bsize); // check to make sure this us valid while I remove the template param
  assert(in_data.size() == nx * ny * nz);
  // device mem where encoded data is stored
  // allocate in encode
  thrust::device_vector<Word> d_encoded;
  thrust::device_vector<double> d_in_data(in_data); 

  // TODO: this does not need to be here, ie, this sets up no
  //       information we can't figure out on the fly
  ConstantSetup::setup_3d(double() , bsize);

  encode<long long int, unsigned long long, double, 64>(dims, d_in_data, d_encoded, bsize, 64); 
  errors.chk("Encode");
  encoded_data.m_data.resize(d_encoded.size());

  Word * d_ptr = thrust::raw_pointer_cast(d_encoded.data());
  Word * h_ptr = &encoded_data.m_data[0];

  cudaMemcpy(h_ptr, d_ptr, d_encoded.size() * sizeof(Word), cudaMemcpyDeviceToHost);

  // set the actual dims and padded dims
  encoded_data.m_dims[0] = nx;
  encoded_data.m_dims[1] = ny;
  encoded_data.m_dims[2] = nz;

  
  
}

void decode(const EncodedData &encoded_data, std::vector<double> &out_data)
{
  const unsigned int bsize = encoded_data.m_bsize;

  int3 dims = make_int3(encoded_data.m_dims[0],
                        encoded_data.m_dims[1],
                        encoded_data.m_dims[2]);

  const size_t out_size = dims.x * dims.y * dims.z;

  thrust::device_vector<double> d_out_data(out_size); 
  thrust::device_vector<Word> d_encoded(encoded_data.m_data);

  ConstantSetup::setup_3d(double() , bsize);

  decode<long long int, unsigned long long, double, 64>(dims, d_encoded, d_out_data, bsize); 

  out_data.resize(out_size); 
  thrust::copy(d_out_data.begin(), 
               d_out_data.end(),
               out_data.begin());
}


} // namespace cuZFP

