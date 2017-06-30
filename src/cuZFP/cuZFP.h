#ifndef cuZFP_h
#define cuZFP_h

#include<vector>

typedef unsigned long long Word;


namespace cuZFP {

struct EncodedData
{
  int                m_dims[3];
  int                m_bsize;
  std::vector<Word>  m_data;
  EncodedData()
    : m_bsize(8) // default rate 
  {
    m_dims[0] = 0;
    m_dims[1] = 0;
    m_dims[2] = 0;
  }
};

extern "C"
void encode_float64(int nx, int ny, int nz, std::vector<double> &in_data, EncodedData  &encoded_data);

extern "C"
void encode_float32(int nx, int ny, int nz, std::vector<float> &in_data, EncodedData  &encoded_data);

extern "C"
void decode_float64(const EncodedData &encoded_data, std::vector<double> &out_data);

extern "C"
void decode_float32(const EncodedData &encoded_data, std::vector<float> &out_data);

} // namespace cuZFP


#endif
