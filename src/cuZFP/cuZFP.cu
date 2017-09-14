#include <assert.h>
#include "cuZFP.h"
#include "encode.cuh"
#include "encode1.cuh"
#include "ErrorCheck.h"
#include "decode.cuh"
#include <constant_setup.cuh>
#include <thrust/device_vector.h>
#include <iostream>

namespace cuZFP {
namespace internal {

template<typename T>
void encode(int nx, int ny, int nz, std::vector<T> &in_data, EncodedData &encoded_data)
{

  ErrorCheck errors;
   
  int3 dims = make_int3(nx, ny, nz);
  const int bsize = encoded_data.m_bsize;

  assert(in_data.size() == nx * ny * nz);
  // device mem where encoded data is stored
  // allocate in encode
  thrust::device_vector<Word> d_encoded;
  thrust::device_vector<T> d_in_data(in_data); 

  ConstantSetup::setup_3d();

  cuZFP::encode<T>(dims, d_in_data, d_encoded, bsize); 

  errors.chk("Encode");
  encoded_data.m_data.resize(d_encoded.size());

  Word * d_ptr = thrust::raw_pointer_cast(d_encoded.data());
  Word * h_ptr = &encoded_data.m_data[0];

  // copy the decoded data back to the host
  cudaMemcpy(h_ptr, d_ptr, d_encoded.size() * sizeof(Word), cudaMemcpyDeviceToHost);

  // set the actual dims and padded dims
  encoded_data.m_dims[0] = nx;
  encoded_data.m_dims[1] = ny;
  encoded_data.m_dims[2] = nz;
}

template<typename T>
void encode(int nx, std::vector<T> &in_data, EncodedData &encoded_data)
{

  ErrorCheck errors;
   
  int dim = nx;
  const int bsize = encoded_data.m_bsize;

  assert(in_data.size() == nx);
  // device mem where encoded data is stored
  // allocate in encode
  thrust::device_vector<Word> d_encoded;
  thrust::device_vector<T> d_in_data(in_data); 
  
  std::cout<<"setting up constants\n";
  ConstantSetup::setup_1d();

  std::cout<<"calling encode\n";
  cuZFP::encode1<T>(dim, d_in_data, d_encoded, bsize); 

  errors.chk("encode1");
  encoded_data.m_data.resize(d_encoded.size());

  Word * d_ptr = thrust::raw_pointer_cast(d_encoded.data());
  Word * h_ptr = &encoded_data.m_data[0];

  // copy the decoded data back to the host
  cudaMemcpy(h_ptr, d_ptr, d_encoded.size() * sizeof(Word), cudaMemcpyDeviceToHost);

  // set the actual dims and padded dims
  encoded_data.m_dims[0] = nx;
  encoded_data.m_dims[1] = 0;
  encoded_data.m_dims[2] = 0;
}

template<typename T>
void decode(const EncodedData &encoded_data, std::vector<T> &out_data)
{

  const unsigned int bsize = encoded_data.m_bsize;

  int3 dims = make_int3(encoded_data.m_dims[0],
                        encoded_data.m_dims[1],
                        encoded_data.m_dims[2]);

  const size_t out_size = dims.x * dims.y * dims.z;

  thrust::device_vector<T> d_out_data(out_size); 
  thrust::device_vector<Word> d_encoded(encoded_data.m_data);

  ConstantSetup::setup_3d();

  cuZFP::decode<T>(dims, d_encoded, d_out_data, bsize); 

  out_data.resize(out_size); 
  thrust::copy(d_out_data.begin(), 
               d_out_data.end(),
               out_data.begin());
}

} // namespace internal

// 3D encoding
void encode(int nx, int ny, int nz, std::vector<double> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, ny, nz, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::f64;
}

void encode(int nx, int ny, int nz, std::vector<float> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, ny, nz, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::f32;
}

void encode(int nx, int ny, int nz, std::vector<int> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, ny, nz, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::i32;
}

void encode(int nx, int ny, int nz, std::vector<long long int> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, ny, nz, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::i64;
}

// 1D encoding
void encode(int nx, std::vector<float> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::f32;
}

// 3D decoding
void decode(const EncodedData &encoded_data, std::vector<double> &out_data)
{
  assert(encoded_data.m_value_type = EncodedData::f64);
  internal::decode(encoded_data, out_data);
}

void decode(const EncodedData &encoded_data, std::vector<float> &out_data)
{
  assert(encoded_data.m_value_type = EncodedData::f32);
  internal::decode(encoded_data, out_data);
}

void decode(const EncodedData &encoded_data, std::vector<int> &out_data)
{
  assert(encoded_data.m_value_type = EncodedData::i32);
  internal::decode(encoded_data, out_data);
}

void decode(const EncodedData &encoded_data, std::vector<long long int> &out_data)
{
  assert(encoded_data.m_value_type = EncodedData::i64);
  internal::decode(encoded_data, out_data);
}


} // namespace cuZFP

