#ifndef cuZFP_h
#define cuZFP_h

#include<vector>

typedef unsigned long long Word;


namespace cuZFP {

struct EncodedData
{
  int                m_dims[3];
  std::vector<Word>  m_data;
  int                m_rate;
  EncodedData()
  {
    m_dims[0] = 0;
    m_dims[1] = 0;
    m_dims[2] = 0;
  }
};

extern "C"
void encode(int nx, int ny, int nz, std::vector<double> &in_data, EncodedData  &encoded_data);

extern "C"
void decode(const EncodedData &encoded_data, std::vector<double> &out_data);

} // namespace cuZFP


#endif
