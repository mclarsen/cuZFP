#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdlib.h>

template<typename T>
void run_test(int nx, int ny, int nz)
{
  const int size = nx * ny * nz;
  std::vector<T> test_data;
  test_data.resize(size);

  for(int z = 0; z < nz; ++z)
    for(int y = 0; y < ny; ++y)
      for(int x = 0; x < nx; ++x)
  {

    T val = static_cast<T>(sqrt(double(z*z) + double(x*x) + double(y*y)));
    if(val != 0.)
    {
      val =  1. / val;
    }
    else
    {
      val = 1.;
    }
    int index = z * nx *ny + y * nx + x;
    test_data[index] = val;
  }

  cuZFP::cu_zfp compressor;
  compressor.set_rate(1);
  compressor.set_field(&test_data[0], cuZFP::get_type<T>() );
  compressor.set_field_size_3d(nx, ny, nz); 
  
  compressor.compress();

  compressor.decompress();

  T *test_data_out = (T*) compressor.get_field();

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
//  run_test<double>(256, 256 ,256);
//}

TEST(encode_decode, test_encode_decode_float32)
{
  //run_test<float>(512, 512, 512);
  run_test<float>(128, 128, 128);
}
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
