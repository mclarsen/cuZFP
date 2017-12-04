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
/*
TEST(sanity_check_float64, test_sanity_check_float64)
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 4;
  int y = 4;
  int z = 8;
  const int size = x * y * z;

  std::vector<double> test_data;
  test_data.resize(size);

  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  cuZFP::EncodedData encoded_data;
  encoded_data.m_bsize = 7;
  cuZFP::encode(x,y,z,test_data, encoded_data);
  dump_raw_binary(encoded_data);
  std::vector<double> test_out_data;
  cuZFP::decode(encoded_data, test_out_data);

  for(int i = 0; i < size; ++i)
  {
    ASSERT_TRUE(i == static_cast<int>(test_out_data.at(i)));
  }
}
*/
TEST(sanity_check_float32, test_sanity_check_float32)
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 4;
  int y = 4;
  int z = 4;
  const int size = x * y * z;
  std::vector<float> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode(x,y,z,test_data, encoded_data);
  std::vector<float> test_out_data;
  cuZFP::decode(encoded_data, test_out_data);

  for(int i = 0; i < size; ++i)
  {
    ASSERT_TRUE(i == static_cast<int>(test_out_data.at(i)));
  }
}
/*
TEST(sanity_check_int32, test_sanity_check_int32)
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 4;
  int y = 4;
  int z = 5;
  const int size = x * y * z;
  std::vector<int> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode(x,y,z,test_data, encoded_data);
  std::vector<int> test_out_data;
  //dump_raw_binary(encoded_data);
  cuZFP::decode(encoded_data, test_out_data);
  
  for(int i = 0; i < size; ++i)
  {
    ASSERT_NEAR(i , static_cast<int>(test_out_data.at(i)), 1);
  }
}

TEST(sanity_check_int64, test_sanity_check_int64)
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 4;
  int y = 4;
  int z = 5;
  const int size = x * y * z;
  std::vector<long long int> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode(x,y,z,test_data, encoded_data);
  std::vector<long long int> test_out_data;
  //dump_raw_binary(encoded_data);
  cuZFP::decode(encoded_data, test_out_data);
  
  for(int i = 0; i < size; ++i)
  {
    ASSERT_NEAR(i , static_cast<int>(test_out_data.at(i)), 1);
  }
}
*/
