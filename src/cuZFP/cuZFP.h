#ifndef cuZFP_h
#define cuZFP_h

#include <stdio.h>

typedef unsigned long long Word;


namespace cuZFP {

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

static size_t
type_size(cuZFP::ValueType type)
{
  if(type == cuZFP::i32) return sizeof(int);
  if(type == cuZFP::i64) return sizeof(long long int);
  if(type == cuZFP::f32) return sizeof(float);
  if(type == cuZFP::f64) return sizeof(double);
  return 0;
}

static void
print_type(cuZFP::ValueType type)
{
  if(type == cuZFP::i32) printf("type: int32\n");
  if(type == cuZFP::i64) printf("type: int64\n");
  if(type == cuZFP::f32) printf("type: float32\n");
  if(type == cuZFP::f64) printf("type: float64\n");
  if(type == cuZFP::none_type) printf("type: none\n");
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
  
  void set_stream(Word *stream, size_t stream_bytes, ValueType type); 

  Word* get_stream();

  size_t get_stream_bytes();

  void decompress();

};

} // namespace cuZFP


#endif
