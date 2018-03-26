#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <cstdio>


using namespace cuZFP;

TEST(sanity_check_1_float32, test_sanity_check_1_float32)
{
  //
  // this test is a simple sanity check to see if
  // we can actually encode and decode with block size.
  // that is not a multiple of four.
  //
  int x = 128;
  const int size = x;
  std::vector<float> test_data;
  test_data.resize(size);
  for(int i = 0; i < size; ++i)
  {
    test_data[i] = i; 
  }

  zfp_stream zfp;  
  zfp_field *field;  

  field = zfp_field_1d(&test_data[0], 
                       zfp_type_float,
                       x);
  
  int rate = 8;

  stream_set_rate(&zfp, rate, field->type, 1);

  size_t buffsize = zfp_stream_maximum_size(&zfp, field);
  unsigned char* buffer = new unsigned char[buffsize];
  zfp.stream = (Word*) buffer;
  compress(&zfp, field);

  std::vector<float> test_data_out;
  test_data_out.resize(size);

  zfp_field *out_field;  

  out_field = zfp_field_1d(&test_data_out[0], 
                           zfp_type_float,
                           x);

  decompress(&zfp, out_field);

  for(int i = 0; i < size; ++i)
  {
    ASSERT_TRUE(i == static_cast<int>(test_data_out[i]));
  }


  zfp_field_free(out_field);
  zfp_field_free(field);
  delete[] buffer;

}

