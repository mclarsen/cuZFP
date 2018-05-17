#include <assert.h>
#include "cuZFP.h"

#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"

#include "ErrorCheck.h"

#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

#include <constant_setup.cuh>
#include <thrust/device_vector.h>
#include <iostream>
#include <type_info.cuh>

namespace cuZFP {
namespace internal {

template<typename T>
size_t encode(int dims[3], int bits_per_block, T *in_data, Word *stream)
{

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
  size_t stream_size = 0;
  if(d == 1)
  {
    int dim = dims[0];
    ConstantSetup::setup_1d();
    stream_size = cuZFP::encode1<T>(dim, d_in_data, d_encoded, bits_per_block); 
  }
  else if(d == 2)
  {
    int2 ndims = make_int2(dims[0], dims[1]);
    ConstantSetup::setup_2d();
    stream_size = cuZFP::encode2<T>(ndims, d_in_data, d_encoded, bits_per_block); 
  }
  else if(d == 3)
  {
    int3 ndims = make_int3(dims[0], dims[1], dims[2]);
    ConstantSetup::setup_3d();
    stream_size = cuZFP::encode<T>(ndims, d_in_data, d_encoded, bits_per_block); 
  }
  errors.chk("Encode");

  Word * d_ptr = thrust::raw_pointer_cast(d_encoded.data());

  size_t stream_bytes = d_encoded.size() * sizeof(Word);
  // copy the decoded data back to the host
  cudaMemcpy(stream, d_ptr, stream_bytes, cudaMemcpyDeviceToHost);
  std::cout<<"Stream size "<<stream_size<<"\n";
  return stream_size; 
}

template<typename T>
void decode(int ndims[3], int bits_per_block, Word *stream, size_t stream_bytes, T *&out)
{

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
  //out = new T[out_size];
  thrust::device_vector<T> d_out_data(out_size); 

  size_t stream_len = stream_bytes / sizeof(Word); 
  thrust::device_vector<Word> d_encoded(stream, stream + stream_len);

  if(d == 3)
  {
    int3 dims = make_int3(ndims[0],
                          ndims[1],
                          ndims[2]);

    ConstantSetup::setup_3d();

    cuZFP::decode<T>(dims, d_encoded, d_out_data, bits_per_block); 

  }
  else if(d == 1)
  {

    int dim = ndims[0];

    ConstantSetup::setup_1d();

    cuZFP::decode1<T>(dim, d_encoded, d_out_data, bits_per_block); 

  }
  else if(d == 2)
  {

    int2 dims;
    dims.x = ndims[0];
    dims.y = ndims[1];

    ConstantSetup::setup_2d();

    cuZFP::decode2<T>(dims, d_encoded, d_out_data, bits_per_block); 

  }
  else std::cout<<" d ==  "<<d<<" not implemented\n";
  
  thrust::copy(d_out_data.begin(), 
               d_out_data.end(),
               out);
}

} // namespace internal

size_t
compress(zfp_stream *stream, zfp_field *field)
{
  int dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;
  size_t stream_bytes = 0;
  if(field->type == zfp_type_float)
  {
    float* data = (float*) field->data;
    stream_bytes = internal::encode<float>(dims, (int)stream->maxbits, data, stream->stream);
  }
  else if(field->type == zfp_type_double)
  {
    double* data = (double*) field->data;
    stream_bytes = internal::encode<double>(dims, (int)stream->maxbits, data, stream->stream);
  }
  else if(field->type == zfp_type_int32)
  {
    int * data = (int*) field->data;
    stream_bytes = internal::encode<int>(dims, (int)stream->maxbits, data, stream->stream);
  }
  else if(field->type == zfp_type_int64)
  {
    long long int * data = (long long int*) field->data;
    stream_bytes = internal::encode<long long int>(dims, (int)stream->maxbits, data, stream->stream);
  }
  return stream_bytes;
}
  
void 
decompress(zfp_stream *stream, zfp_field *field)
{
  int dims[3];
  dims[0] = field->nx;
  dims[1] = field->ny;
  dims[2] = field->nz;

  //TODO this will be changed
  
  size_t bytes = zfp_stream_maximum_size(stream, field);

  if(field->type == zfp_type_float)
  {
    float *data = (float*) field->data;
    internal::decode(dims, (int)stream->maxbits, stream->stream, bytes, data);
    field->data = (void*) data;
  }
  else if(field->type == zfp_type_double)
  {
    double *data = (double*) field->data;
    internal::decode(dims, (int)stream->maxbits, stream->stream, bytes, data);
    field->data = (void*) data;
  }
  else if(field->type == zfp_type_int32)
  {
    int *data = (int*) field->data;
    internal::decode(dims, (int)stream->maxbits, stream->stream, bytes, data);
    field->data = (void*) data;
  }
  else if(field->type == zfp_type_int64)
  {
    long long int *data = (long long int*) field->data;
    internal::decode(dims, (int)stream->maxbits, stream->stream, bytes, data);
    field->data = (void*) data;
  }
  else
  {
    std::cerr<<"Cannot decompress: type unknown\n";
  }
}

} // namespace cuZFP

