#ifndef cuZFP_h
#define cuZFP_h

#include<vector>

typedef unsigned long long Word;


namespace cuZFP {

struct EncodedData
{
  enum ValueType     { f32, f64, i32, i64 };
  int                m_dims[3];
  int                m_bsize;
  ValueType          m_value_type;
  std::vector<Word>  m_data;
  EncodedData()
    : m_bsize(8) // default rate 
  {
    m_dims[0] = 0;
    m_dims[1] = 0;
    m_dims[2] = 0;
  }
};

// 3D encoding 
void encode(int nx, int ny, int nz, std::vector<double> &in_data, EncodedData  &encoded_data);

void encode(int nx, int ny, int nz, std::vector<float> &in_data, EncodedData  &encoded_data);

void encode(int nx, int ny, int nz, std::vector<int> &in_data, EncodedData  &encoded_data);

void encode(int nx, int ny, int nz, std::vector<long long int> &in_data, EncodedData  &encoded_data);

// 1D encoding
void encode(int nx, std::vector<float> &in_data, EncodedData  &encoded_data);

// 3D decoding 
void decode(const EncodedData &encoded_data, std::vector<double> &out_data);

void decode(const EncodedData &encoded_data, std::vector<float> &out_data);

void decode(const EncodedData &encoded_data, std::vector<int> &out_data);

void decode(const EncodedData &encoded_data, std::vector<long long int> &out_data);

} // namespace cuZFP


#endif
