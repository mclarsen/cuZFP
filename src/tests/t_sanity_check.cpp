#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>

TEST(sanity_check, test_sanity_check)
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

  std::vector<double> test_out_data;
  cuZFP::decode(encoded_data, test_out_data);

  for(int i = 0; i < size; ++i)
  {
    ASSERT_TRUE(i == static_cast<int>(test_out_data.at(i)));
  }
}

