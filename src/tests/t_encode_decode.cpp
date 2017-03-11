
#include<gtest/gtest.h>
#include <cuZFP.h>
#include<vector>

TEST(encode_decode, test_encode_decode)
{
  std::vector<double> test_data;
  test_data.resize(64);
  std::vector<int> encoded_data;

  cuZFP::encode_vector(test_data, encoded_data);
}

