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

