#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdlib.h>

TEST(encode_decode_rand, test_encode_decode_rand)
{
  int nx = 256;
  int ny = 256;
  int nz = 256;
  const int size = nx * ny * nz;
  std::vector<double> test_data;
  test_data.resize(size);

  for(int z = 0; z < nz; ++z)
    for(int y = 0; y < ny; ++y)
      for(int x = 0; x < nx; ++x)
  {

    double val = rand() / 1000000000.;
    int index = z * nx *ny + y * nx + x;
    test_data[index] = val;
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode_float64(nx,ny,nz,test_data, encoded_data);


  std::vector<double> test_data_out;
  cuZFP::decode_float64(encoded_data, test_data_out);
  double tot_err = 0;
  for(int i = 0; i < size; ++i)
  {
      tot_err += abs(test_data_out[i] - test_data[i]);
  }

  double average_err = tot_err /  double(size);
  printf("Total absolute error %2.20f\n", tot_err);
  printf("Average abosulte error %2.20f with %d values.\n", average_err, size);
}

