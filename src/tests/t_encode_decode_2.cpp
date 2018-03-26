#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdlib.h>

using namespace cuZFP;

template<typename T>
void run_test(int nx, int ny)
{
  const int size = nx * ny;
  std::vector<T> test_data;
  test_data.resize(size);

  for(int y = 0; y < ny; ++y)
    for(int x = 0; x < nx; ++x)
    {
      double v = sqrt(double(x) * double(x) + double(y) * double(y));
      T val = static_cast<T>(v);
      int index = y * nx + x;
      test_data[index] = val;
    }

  zfp_stream zfp;  
  zfp_field *field;  
  zfp_type type = get_zfp_type<T>();

  field = zfp_field_2d(&test_data[0], 
                       type,
                       nx,
                       ny);
  
  int rate = 8;

  double actual_rate = stream_set_rate(&zfp, rate, type, 2);
  std::cout<<"actual rate "<<actual_rate<<"\n"; 

  size_t buffsize = zfp_stream_maximum_size(&zfp, field);
  unsigned char* buffer = new unsigned char[buffsize];
  zfp.stream = (Word*) buffer;
  compress(&zfp, field);

  std::vector<float> test_data_out;
  test_data_out.resize(size);

  zfp_field *out_field;  

  out_field = zfp_field_2d(&test_data_out[0], 
                           type,
                           nx,
                           ny);

  decompress(&zfp, out_field);

  zfp_field_free(out_field);
  zfp_field_free(field);
  delete[] buffer;

  double tot_err = 0;
  for(int i = 0; i < size; ++i)
  {
      tot_err += abs(test_data_out[i] - test_data[i]);
  }

  double average_err = tot_err /  double(size);
  printf("Total absolute error %2.20f\n", tot_err);
  printf("Average abosulte error %2.20f with %d values.\n", average_err, size);
}


TEST(encode_decode, test_encode_decode_float32)
{
  run_test<float>(1024,1024);
}
