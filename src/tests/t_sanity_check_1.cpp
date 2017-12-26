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

template<typename T>
void dump_decoded(std::vector<T> &data)
{

  int n = data.size(); 

  for(int i = 0; i < n; i++)
  {
    fwrite(&data[i], sizeof(T), 1, stderr);
  }
}


TEST(sanity_check_1_float32, test_sanity_check_1_float32)
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 128;
  const int size = x;
  std::vector<int> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }
  
  cuZFP::EncodedData encoded_data;
  encoded_data.m_bsize = 8; // 2 blocks per word
  cuZFP::encode(x, test_data, encoded_data);
  std::vector<int> test_out_data;
  dump_raw_binary(encoded_data);
  cuZFP::decode(encoded_data, test_out_data);
  //dump_decoded(test_out_data);
  for(int i = 0; i < size; ++i)
  {
     //std::cout<<test_out_data.at(i)<<"\n";
     //ASSERT_TRUE(i == static_cast<int>(test_out_data.at(i)));
  }
}

/*
TEST(sanity_check_1_float64, test_sanity_check_1_float64)
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 256;
  const int size = x;
  std::vector<double> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }
  
  cuZFP::EncodedData encoded_data;
  encoded_data.m_bsize = 8; // 2 blocks per word
  cuZFP::encode(x, test_data, encoded_data);
  std::vector<double> test_out_data;
  //dump_raw_binary(encoded_data);
  //cuZFP::decode(encoded_data, test_out_data);

  for(int i = 0; i < size; ++i)
  {
   // ASSERT_TRUE(i == static_cast<int>(test_out_data.at(i)));
  }
}
TEST(sanity_check_1_int64, test_sanity_check_1_int64)
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 128;
  const int size = x;
  std::vector<long long int> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }
  
  cuZFP::EncodedData encoded_data;
  //encoded_data.m_bsize = 8; // 2 blocks per word
  encoded_data.m_bsize = 3; // 2 blocks per word
  cuZFP::encode(x, test_data, encoded_data);
  std::vector<long long int> test_out_data;
  dump_raw_binary(encoded_data);
  //cuZFP::decode(encoded_data, test_out_data);

  for(int i = 0; i < size; ++i)
  {
   // ASSERT_TRUE(i == static_cast<int>(test_out_data.at(i)));
  }
}
*/
