#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <cstdio>

void dump_raw_binary(cuZFP::EncodedData &data)
{

  int n = data.m_data.size(); 

  for(int i = 0; i < n; i++)
  {
    fwrite(&data.m_data[i], sizeof(Word), 1, stderr);
  }
}

TEST(sanity_check_float64, test_sanity_check_float64)
{
  //
  // this test is a simple sanity checjk to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 4;
  int y = 4;
  int z = 5;
  const int size = x * y * z;

  std::vector<double> test_data;
  test_data.resize(size);

  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode_float64(x,y,z,test_data, encoded_data);

  std::vector<double> test_out_data;
  cuZFP::decode_float64(encoded_data, test_out_data);

  for(int i = 0; i < size; ++i)
  {
    ASSERT_TRUE(i == static_cast<int>(test_out_data.at(i)));
  }
}

TEST(sanity_check_float32, test_sanity_check_float32)
{
  //
  // this test is a simple sanity checjk to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 4;
  int y = 4;
  int z = 5;
  const int size = x * y * z;
  std::vector<float> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode_float32(x,y,z,test_data, encoded_data);
  std::vector<float> test_out_data;
  cuZFP::decode_float32(encoded_data, test_out_data);

  for(int i = 0; i < size; ++i)
  {
    //std::cout<<i<<" "<<test_out_data.at(i)<<"\n";
    ASSERT_TRUE(i == static_cast<int>(test_out_data.at(i)));
  }
}
