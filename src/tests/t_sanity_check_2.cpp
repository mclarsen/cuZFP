#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <cstdio>

template<typename T>
void dump_raw_binary(std::vector<T> &data)
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
  int x = 4;
  int y = 4;
  const int size = x * y;
  std::vector<float> test_data;
  test_data.resize(size);

  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }
  
  cuZFP::cu_zfp compressor;
  compressor.set_rate(8);
  compressor.set_field(&test_data[0], cuZFP::get_type<float>() );
  compressor.set_field_size_2d(x, y); 
  
  compressor.compress();

  compressor.decompress();

  float *test_data_out = (float*) compressor.get_field();

  for(int i = 0; i < size; ++i)
  {
    ASSERT_TRUE(i == static_cast<int>(test_data_out[i]));
  }
}

