#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdlib.h>

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
      test_data[x] = val;
    }

  cuZFP::EncodedData encoded_data;
  encoded_data.m_bsize = 4;
  cuZFP::encode(nx, ny, test_data, encoded_data);

  //std::vector<T> test_data_out;
  //cuZFP::decode(encoded_data, test_data_out);
  //double tot_err = 0;
  //for(int i = 0; i < size; ++i)
  //{
  //    tot_err += abs(test_data_out[i] - test_data[i]);
  //}

  //double average_err = tot_err /  double(size);
  //printf("Total absolute error %2.20f\n", tot_err);
  //printf("Average abosulte error %2.20f with %d values.\n", average_err, size);
}


TEST(encode_decode, test_encode_decode_float32)
{
  run_test<float>(2048, 2048);
}
