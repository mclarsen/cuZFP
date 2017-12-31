#ifndef cuZFP_h
#define cuZFP_h

#include<vector>

typedef unsigned long long Word;


namespace cuZFP {

//struct EncodedData
//{
//  enum ValueType     { f32, f64, i32, i64 };
//  int                m_dims[3];
//  int                m_bsize;
//  ValueType          m_value_type;
//  std::vector<Word>  m_data;
//  EncodedData()
//    : m_bsize(8) // default rate 
//  {
//    m_dims[0] = 0;
//    m_dims[1] = 0;
//    m_dims[2] = 0;
//  }
//};

enum ValueType { f32, f64, i32, i64, none_type };

template<typename T> 
inline ValueType get_type()
{
  return none_type;
}

template<> 
inline ValueType get_type<float>()
{
  return f32;
}

template<> 
inline ValueType get_type<double>()
{
  return f64;
}

template<> 
inline ValueType get_type<int>()
{
  return i32;
}

template<> 
inline ValueType get_type<long long int>()
{
  return i64;
}

class cu_zfp 
{
protected:
  int          m_dims[3];
  int          m_rate;
  ValueType    m_value_type;

  Word        *m_stream;
  void        *m_field;
  bool         m_owns_field;
  bool         m_owns_stream;
  size_t       m_stream_bytes;
public:
  cu_zfp();
  ~cu_zfp();

  void set_field_size_1d(int nx);

  void set_field_size_2d(int nx, int ny);

  void set_field_size_3d(int nx, int ny, int nz);

  void set_field(void *field, ValueType type);
  
  void set_rate(int rate);
  int get_rate();

  void* get_field();

  ValueType get_field_type();

  void compress();
  
  void set_stream(Word *stream, size_t stream_bytes); 

  Word* get_stream();

  size_t get_stream_bytes();

  void decompress();

};

// --------------------------- 3D encoding --------------------------------
//void encode(int nx, int ny, int nz, std::vector<double> &in_data, EncodedData  &encoded_data);
//
//void encode(int nx, int ny, int nz, std::vector<float> &in_data, EncodedData  &encoded_data);
//
//void encode(int nx, int ny, int nz, std::vector<int> &in_data, EncodedData  &encoded_data);
//
//void encode(int nx, int ny, int nz, std::vector<long long int> &in_data, EncodedData  &encoded_data);
//
//// --------------------------- 1D encoding --------------------------------
//void encode(int nx, std::vector<float> &in_data, EncodedData  &encoded_data);
//
//void encode(int nx, std::vector<double> &in_data, EncodedData  &encoded_data);
//
//void encode(int nx, std::vector<int> &in_data, EncodedData  &encoded_data);
//
//void encode(int nx, std::vector<long long int> &in_data, EncodedData  &encoded_data);
//
//// --------------------------- 2D encoding --------------------------------
//void encode(int nx, int ny, std::vector<float> &in_data, EncodedData  &encoded_data);
//
//// --------------------------- 3D decoding --------------------------------
//void decode(const EncodedData &encoded_data, std::vector<double> &out_data);
//
//void decode(const EncodedData &encoded_data, std::vector<float> &out_data);
//
//void decode(const EncodedData &encoded_data, std::vector<int> &out_data);
//
//void decode(const EncodedData &encoded_data, std::vector<long long int> &out_data);

} // namespace cuZFP


#endif
