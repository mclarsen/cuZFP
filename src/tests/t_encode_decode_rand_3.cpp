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

  cuZFP::cu_zfp compressor;
  compressor.set_rate(1);
  compressor.set_field(&test_data[0], cuZFP::get_type<double>() );
  compressor.set_field_size_3d(nx, ny, nz); 
  
  compressor.compress();

  compressor.decompress();

  double *test_data_out = (double*) compressor.get_field();

  double tot_err = 0;
  for(int i = 0; i < size; ++i)
  {
      tot_err += abs(test_data_out[i] - test_data[i]);
  }

  double average_err = tot_err /  double(size);
  printf("Total absolute error %2.20f\n", tot_err);
  printf("Average abosulte error %2.20f with %d values.\n", average_err, size);
}

