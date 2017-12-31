#include <assert.h>
#include "cuZFP.h"
#include "encode.cuh"
#include "encode1.cuh"
#include "encode2.cuh"
#include "ErrorCheck.h"
#include "decode.cuh"
#include "decode1.cuh"
#include "decode2.cuh"
#include <constant_setup.cuh>
#include <thrust/device_vector.h>
#include <iostream>

namespace cuZFP {
namespace internal {

template<typename T>
void encode(int dims[3], int rate, T *in_data, Word *&stream, size_t &stream_bytes)
{

  
  const int bsize = rate;
  int d = 0;
  size_t len = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(dims[i] != 0)
    {
      d++;
      len *= dims[i];
    }
  }
   
  // allocate in encode
  thrust::device_vector<Word> d_encoded;
  thrust::device_vector<T> d_in_data(in_data, in_data + len); 

  ErrorCheck errors;
  if(d == 1)
  {
    int dim = dims[0];
    ConstantSetup::setup_1d();
    cuZFP::encode1<T>(dim, d_in_data, d_encoded, bsize); 
  }
  else if(d == 2)
  {
    int2 ndims = make_int2(dims[0], dims[1]);
    ConstantSetup::setup_2d();
    cuZFP::encode2<T>(ndims, d_in_data, d_encoded, bsize); 
  }
  else if(d == 3)
  {
    int3 ndims = make_int3(dims[0], dims[1], dims[2]);
    ConstantSetup::setup_3d();
    cuZFP::encode<T>(ndims, d_in_data, d_encoded, bsize); 
  }
  errors.chk("Encode");



  stream = new Word[d_encoded.size()];

  Word * d_ptr = thrust::raw_pointer_cast(d_encoded.data());

  stream_bytes = d_encoded.size() * sizeof(Word);
  // copy the decoded data back to the host
  cudaMemcpy(stream, d_ptr, stream_bytes, cudaMemcpyDeviceToHost);
  
}

template<typename T>
void decode(int ndims[3], int rate, Word *stream, size_t stream_bytes, T *&out)
{

  const unsigned int bsize = rate ;

  int d = 0;
  size_t out_size = 1;
  for(int i = 0; i < 3; ++i)
  {
    if(ndims[i] != 0)
    {
      d++;
      out_size *= ndims[i];
    }
  }

  //allocate space
  out = new T[out_size];
  thrust::device_vector<T> d_out_data(out_size); 

  size_t stream_len = stream_bytes / sizeof(Word); 
  thrust::device_vector<Word> d_encoded(stream, stream + stream_len);

  if(d == 3)
  {
    int3 dims = make_int3(ndims[0],
                          ndims[1],
                          ndims[2]);

    ConstantSetup::setup_3d();

    cuZFP::decode<T>(dims, d_encoded, d_out_data, bsize); 

  }
  else if(d == 1)
  {

    int dim = ndims[0];

    ConstantSetup::setup_1d();

    cuZFP::decode1<T>(dim, d_encoded, d_out_data, bsize); 

  }
  else if(d == 2)
  {

    int2 dims;
    dims.x = ndims[0];
    dims.y = ndims[1];

    ConstantSetup::setup_2d();

    cuZFP::decode2<T>(dims, d_encoded, d_out_data, bsize); 

  }
  else std::cout<<" d ==  "<<d<<" not implemented\n";
  
  thrust::copy(d_out_data.begin(), 
               d_out_data.end(),
               out);
}

#if 0
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
  
  ConstantSetup::setup_1d();

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
void encode(int nx, int ny, std::vector<T> &in_data, EncodedData &encoded_data)
{

  ErrorCheck errors;
   
  int2 dims;
  dims.x = nx;
  dims.y = ny;
  const int bsize = encoded_data.m_bsize;

  assert(in_data.size() == nx * ny);
  // device mem where encoded data is stored
  // allocate in encode
  thrust::device_vector<Word> d_encoded;
  thrust::device_vector<T> d_in_data(in_data); 
  
  std::cout<<"setting up constants\n";
  ConstantSetup::setup_2d();

  std::cout<<"calling encode\n";
  cuZFP::encode2<T>(dims, d_in_data, d_encoded, bsize); 

  errors.chk("encode2");
  encoded_data.m_data.resize(d_encoded.size());

  Word * d_ptr = thrust::raw_pointer_cast(d_encoded.data());
  Word * h_ptr = &encoded_data.m_data[0];

  // copy the decoded data back to the host
  cudaMemcpy(h_ptr, d_ptr, d_encoded.size() * sizeof(Word), cudaMemcpyDeviceToHost);

  // set the actual dims and padded dims
  encoded_data.m_dims[0] = nx;
  encoded_data.m_dims[1] = ny;
  encoded_data.m_dims[2] = 0;
}

template<typename T>
void decode(const EncodedData &encoded_data, std::vector<T> &out_data)
{

  const unsigned int bsize = encoded_data.m_bsize;

  int d = 0;
  for(int i = 0; i < 3; ++i)
  {
    if(encoded_data.m_dims[i] != 0) d++;
  }
  if(d == 3)
  {
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
  else if(d == 1)
  {

    int dim = encoded_data.m_dims[0];
    const size_t out_size = dim;

    thrust::device_vector<T> d_out_data(out_size); 
    thrust::device_vector<Word> d_encoded(encoded_data.m_data);

    ConstantSetup::setup_1d();

    cuZFP::decode1<T>(dim, d_encoded, d_out_data, bsize); 

    out_data.resize(out_size); 
    thrust::copy(d_out_data.begin(), 
                 d_out_data.end(),
                 out_data.begin());
  }
  else if(d == 2)
  {

    int2 dims;
    dims.x = encoded_data.m_dims[0];
    dims.y = encoded_data.m_dims[1];
    const size_t out_size = dims.x * dims.y;

    thrust::device_vector<T> d_out_data(out_size); 
    thrust::device_vector<Word> d_encoded(encoded_data.m_data);

    ConstantSetup::setup_2d();

    cuZFP::decode2<T>(dims, d_encoded, d_out_data, bsize); 

    out_data.resize(out_size); 
    thrust::copy(d_out_data.begin(), 
                 d_out_data.end(),
                 out_data.begin());
  }
  else std::cout<<" d ==  "<<d<<" not implemented\n";
  
}


// -----------------------------  3D encoding -------------------------------------------
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

// -------------------------- 1D encoding --------------------------------------------
void encode(int nx, std::vector<float> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::f32;
}

void encode(int nx, std::vector<double> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::f64;
}

void encode(int nx, std::vector<int> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::i32;
}

void encode(int nx, std::vector<long long int> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::i64;
}

// -------------------------- 2D encoding --------------------------------------------
void encode(int nx, int ny, std::vector<float> &in_data, EncodedData &encoded_data)
{
  internal::encode(nx, ny, in_data, encoded_data);  
  encoded_data.m_value_type = EncodedData::f32;
}

// --------------------------- decoding --------------------------------------------
void decode(const EncodedData &encoded_data, std::vector<double> &out_data)
{
  //assert(encoded_data.m_value_type = EncodedData::f64);
  internal::decode(encoded_data, out_data);
}

void decode(const EncodedData &encoded_data, std::vector<float> &out_data)
{
  //assert(encoded_data.m_value_type = EncodedData::f32);
  internal::decode(encoded_data, out_data);
}

void decode(const EncodedData &encoded_data, std::vector<int> &out_data)
{
  //assert(encoded_data.m_value_type = EncodedData::i32);
  internal::decode(encoded_data, out_data);
}

void decode(const EncodedData &encoded_data, std::vector<long long int> &out_data)
{
  //assert(encoded_data.m_value_type = EncodedData::i64);
  internal::decode(encoded_data, out_data);
}
#endif

} // namespace internal

cu_zfp::cu_zfp()
  : m_rate(8),
    m_stream(NULL),
    m_field(NULL),
    m_owns_field(false),
    m_owns_stream(false),
    m_stream_bytes(0)
{
  m_dims[0] = 0;
  m_dims[1] = 0;
  m_dims[2] = 0;
}

cu_zfp::~cu_zfp()
{
  if(m_field && m_owns_field) 
  {
    std::cout<<"Deleting field "<<m_field<<"\n";
    if(m_value_type == f32)
    {
      float *field = (float*) m_field;
      delete[] field;
    }
    else if(m_value_type == f64)
    {
      double *field = (double*) m_field;
      delete[] field;
    }
    else if(m_value_type == i32)
    {
      int *field = (int*) m_field;
      delete[] field;
    }
    else if(m_value_type == i64)
    {
      long long int *field = (long long int*) m_field;
      delete[] field;
    }
  }

  if(m_stream && m_owns_stream) 
  {
    std::cout<<"Deleting stream\n";
    delete[] m_stream;
  }
}

void 
cu_zfp::set_field_size_1d(int nx)
{
  m_dims[0] = nx;
  m_dims[1] = 0;
  m_dims[2] = 0;
}

void 
cu_zfp::set_field_size_2d(int nx, int ny)
{
  m_dims[0] = nx;
  m_dims[1] = ny;
  m_dims[2] = 0;
}

void 
cu_zfp::set_field_size_3d(int nx, int ny, int nz)
{
  m_dims[0] = nx;
  m_dims[1] = ny;
  m_dims[2] = nz;
}

void 
cu_zfp::set_field(void *field, ValueType type)
{
  m_field = field;
  m_value_type = type;
  m_owns_field = false;
  std::cout<<"setting field "<<m_field<<"\n";
}

void* 
cu_zfp::get_field()
{
  return m_field;
}

ValueType 
cu_zfp::get_field_type()
{
  return m_value_type;
}

void 
cu_zfp::compress()
{
  if(m_field == NULL)
  {
    std::cerr<<"Compress error: field never set\n";
  }
  else
  {
    m_owns_stream = false; 
    if(m_value_type == f32)
    {
      float* field = (float*) m_field;
      internal::encode<float>(m_dims, m_rate, field, m_stream, m_stream_bytes);
    }
    else if(m_value_type == f64)
    {
      double* field = (double*) m_field;
      internal::encode<double>(m_dims, m_rate, field, m_stream, m_stream_bytes);
    }
    else if(m_value_type == i32)
    {
      int * field = (int*) m_field;
      internal::encode<int>(m_dims, m_rate, field, m_stream, m_stream_bytes);
    }
    else if(m_value_type == i64)
    {
      long long int * field = (long long int*) m_field;
      internal::encode<long long int>(m_dims, m_rate, field, m_stream, m_stream_bytes);
    }
  }
}
  
void 
cu_zfp::set_stream(Word *stream, size_t stream_bytes)
{
  m_stream = stream;
  m_stream_bytes = stream_bytes;
  m_owns_stream = false;
}

Word* 
cu_zfp::get_stream()
{
  return m_stream;
}

size_t
cu_zfp::get_stream_bytes()
{
  return m_stream_bytes;
}

void 
cu_zfp::decompress()
{
  if(m_stream == NULL)
  {
    std::cerr<<"Decompress error: stream never set\n";
  }
  else
  {
    m_owns_field = true; 
    if(m_value_type == f32)
    {
      float *field = (float*) m_field;
      internal::decode(m_dims, m_rate, m_stream, m_stream_bytes, field);
      m_field = (void*) field;
    }
    else if(m_value_type == f64)
    {
      double *field = (double*) m_field;
      internal::decode(m_dims, m_rate, m_stream, m_stream_bytes, field);
      m_field = (void*) field;
    }
    else if(m_value_type == i32)
    {
      int *field = (int*) m_field;
      internal::decode(m_dims, m_rate, m_stream, m_stream_bytes, field);
      m_field = (void*) field;
    }
    else if(m_value_type == i64)
    {
      long long int *field = (long long int*) m_field;
      internal::decode(m_dims, m_rate, m_stream, m_stream_bytes, field);
      m_field = (void*) field;
    }
  }
}

void
cu_zfp::set_rate(int rate)
{
  m_rate = rate;
}

int 
cu_zfp::get_rate()
{
  return m_rate;
}

} // namespace cuZFP

