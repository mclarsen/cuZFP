#include <gtest/gtest.h>
#include <cuZFP.h>
#include <vector>
#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdlib.h>


template<typename T>
void dump(std::vector<T> &data)
{

  int n = data.size(); 

  for(int i = 0; i < n; i++)
  {
    fwrite(&data[i], sizeof(T), 1, stderr);
  }
}

template<typename T>
void run_test(int nx)
{
  const int size = nx;
  std::vector<T> test_data;
  test_data.resize(size);

  for(int x = 0; x < nx; ++x)
  {
    double v = double (x) * (3.14/180.);
    T val = static_cast<T>(sin(v)*10.);
    test_data[x] = val;
  }
  //dump(test_data); 
  cuZFP::EncodedData encoded_data;
  encoded_data.m_bsize = 3;
  cuZFP::encode(nx, test_data, encoded_data);
  //dump(encoded_data.m_data); 
  std::vector<T> test_data_out;
  cuZFP::decode(encoded_data, test_data_out);
  dump(test_data_out); 
  double tot_err = 0;
  for(int i = 0; i < size; ++i)
  {
      tot_err += abs(test_data_out[i] - test_data[i]);
      //if(i % 4 == 0) std::cout<<"\n";
      //if(i < 4) std::cout<<"decompressed "<<test_data_out[i]<<" "<<test_data[i]<<"\n";
  }

  double average_err = tot_err /  double(size);
  printf("Total absolute error %2.20f\n", tot_err);
  printf("Average abosulte error %2.20f with %d values.\n", average_err, size);
}

//TEST(encode_decode, test_encode_decode_float64)
//{
//  run_test<int>(128 * 128 * 128);
//}

TEST(encode_decode, test_encode_decode_float64)
{
  //run_test<double>(128 * 128 * 128);
  run_test<float>(256);
}

//TEST(encode_decode, test_encode_decode_float32)
//{
//  run_test<double>(128*128*128);
//}
//
//TEST(encode_decode, test_encode_decode_int64)
//{
//  run_test<long long int>(256, 256, 256);
//}
//
//TEST(encode_decode, test_encode_decode_int32)
//{
//  run_test<int>(512, 512, 512);
//}
