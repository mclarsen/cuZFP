#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#if 1
TEST(encode_decode, test_encode_decode)
{
  int x = 4;
  int y = 4;
  int z = 4;
  const int size = x * y * z;
  std::vector<double> test_data;
  test_data.resize(size);

  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode(x,y,z,test_data, encoded_data);

  for(int i = 0; i < encoded_data.m_data.size(); ++i)
  {
    std::cout<<encoded_data.m_data.at(i)<<"\n";
  }

  std::cout<<"-----------------------------------------------------\n";

  std::vector<double> test_out_data;
  cuZFP::decode(encoded_data, test_out_data);
  std::cout<<"Size "<<size<<"\n";
  for(int i = 0; i < size; ++i)
  {
    std::cout<<"i "<<i<<" "<<test_out_data.at(i)<<"\n";
  }
}
#endif
#if 1
TEST(encode_decode_large, test_encode_decode_large)
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

    double val = sqrt(double(z*z) + double(x*x) + double(y*y));
    val =  1. / val;
    int index = z * nx *ny + y * nx + x;
    test_data[index] = val;
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode(nx,ny,nz,test_data, encoded_data);


  std::vector<double> test_data_out;
  cuZFP::decode(encoded_data, test_data_out);
  double tot_err = 0;
  for(int i = 0; i < size; ++i)
  {
      tot_err += abs((test_data[i] - test_data_out[i]) / test_data[i]);
  }
  std::cout<<std::setprecision(15);
  std::cout<<abs((test_data[100] - test_data_out[100]) / test_data[100])<<"\n";

  tot_err /= double(size);
  std::cout<<"Total error "<<tot_err<<" in "<<size<<" values.\n";
}
#endif

