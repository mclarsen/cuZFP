
#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
TEST(encode_decode, test_encode_decode)
{
  const int size = 64;
  std::vector<double> test_data;
  test_data.resize(size);

  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i;
  }

  cuZFP::EncodedData encoded_data;
  cuZFP::encode(4,4,4,test_data, encoded_data);

  for(int i = 0; i < encoded_data.m_data.size(); ++i)
  {
    std::cout<<encoded_data.m_data.at(i)<<"\n";
  }

  std::cout<<"-----------------------------------------------------\n";

  std::vector<double> test_out_data;
  cuZFP::decode(encoded_data, test_out_data);

  for(int i = 0; i < size; ++i)
  {
    std::cout<<test_out_data.at(i)<<"\n";
  }
}

