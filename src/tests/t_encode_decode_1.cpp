#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdlib.h>


using namespace cuZFP;

template<typename T>
void run_test(int nx)
{
  const int size = nx;
  std::vector<T> test_data;
  test_data.resize(size);

  for(int x = 0; x < nx; ++x)
  {
    double v = double (x) * (3.14/180.);
    T val = static_cast<T>(sin(v)*10.);
    test_data[x] = val;
  }
  
  zfp_stream zfp;  
  zfp_field *field;  
  zfp_type type = get_zfp_type<T>();

  field = zfp_field_1d(&test_data[0], 
                       type,
                       nx);
  
  int rate = 8;

  stream_set_rate(&zfp, rate, type, 1);

  size_t buffsize = zfp_stream_maximum_size(&zfp, field);
  unsigned char* buffer = new unsigned char[buffsize];
  zfp.stream = (Word*) buffer;
  compress(&zfp, field);

  std::vector<float> test_data_out;
  test_data_out.resize(size);

  zfp_field *out_field;  

  out_field = zfp_field_1d(&test_data_out[0], 
                           type,
                           nx);

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

//TEST(encode_decode, test_encode_decode_float64)
//{
//  run_test<int>(128 * 128 * 128);
//}

TEST(encode_decode, test_encode_decode_float64)
{
  //run_test<double>(128 * 128 * 128);
  run_test<float>(256);
}

//TEST(encode_decode, test_encode_decode_float32)
//{
//  run_test<double>(128*128*128);
//}
//
//TEST(encode_decode, test_encode_decode_int64)
//{
//  run_test<long long int>(256, 256, 256);
//}
//
//TEST(encode_decode, test_encode_decode_int32)
//{
//  run_test<int>(512, 512, 512);
//}
