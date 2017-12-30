#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdlib.h>

template<typename T>
void dump_raw_binary(std::vector<T> &data)
{

  int n = data.size(); 

  for(int i = 0; i < n; i++)
  {
    fwrite(&data[i], sizeof(T), 1, stderr);
  }
}

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
  dump_raw_binary(test_data);
  cuZFP::EncodedData encoded_data;
  encoded_data.m_bsize = 1;
  cuZFP::encode(nx, ny, test_data, encoded_data);

  //dump_raw_binary(encoded_data.m_data);
  std::vector<T> test_data_out;
  cuZFP::decode(encoded_data, test_data_out);
  double tot_err = 0;
  for(int i = 0; i < size; ++i)
  {
      tot_err += abs(test_data_out[i] - test_data[i]);
      //std::cout<<"data "<<test_data[i]<<" decomp "<<test_data_out[i]<<"\n";
  }

  //dump_raw_binary(test_data_out);
  double average_err = tot_err /  double(size);
  printf("Total absolute error %2.20f\n", tot_err);
  printf("Average abosulte error %2.20f with %d values.\n", average_err, size);
}


TEST(encode_decode, test_encode_decode_float32)
{
  run_test<float>(1024,1024);
}