#ifndef SHARED_H
#define SHARED_H

#ifndef DBLOCK
  #define DBLOCK -1
#endif

#include <type_info.cuh>
#include <zfp_structs.h>
#include <stdio.h>

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

#define LDEXP(x, e) ldexp(x, e)
#define FREXP(x, e) frexp(x, e)
#define FABS(x) fabs(x)

#define NBMASK 0xaaaaaaaaaaaaaaaaull

__constant__ unsigned char c_perm_1[4];
__constant__ unsigned char c_perm_2[16];
__constant__ unsigned char c_perm[64];

namespace cuZFP
{
// map two's complement signed integer to negabinary unsigned integer
inline __device__ __host__
unsigned long long int int2uint(const long long int x)
{
    return (x + (unsigned long long int)0xaaaaaaaaaaaaaaaaull) ^ 
                (unsigned long long int)0xaaaaaaaaaaaaaaaaull;
}

inline __device__ __host__
unsigned int int2uint(const int x)
{
    return (x + (unsigned int)0xaaaaaaaau) ^ 
                (unsigned int)0xaaaaaaaau;
}

template<class Scalar>
__host__ __device__
static int
exponent(Scalar x)
{
  if (x > 0) {
    int e;
    FREXP(x, &e);
    // clamp exponent in case x is denormalized
    return MAX(e, 1 - get_ebias<Scalar>());
  }
  return -get_ebias<Scalar>();
}


// lifting transform of 4-vector
template <class Int, uint s>
__device__ __host__
static void
fwd_lift(Int* p)
{
  Int x = *p; p += s;
  Int y = *p; p += s;
  Int z = *p; p += s;
  Int w = *p; p += s;

  // default, non-orthogonal transform (preferred due to speed and quality)
  //        ( 4  4  4  4) (x)
  // 1/16 * ( 5  1 -1 -5) (y)
  //        (-4  4  4 -4) (z)
  //        (-2  6 -6  2) (w)
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

template<typename Scalar>
Scalar
inline __device__
quantize_factor(const int &exponent, Scalar);

template<>
float
inline __device__
quantize_factor<float>(const int &exponent, float)
{
	return  LDEXP(1.0, get_precision<float>() - 2 - exponent);
}

template<>
double
inline __device__
quantize_factor<double>(const int &exponent, double)
{
	return  LDEXP(1.0, get_precision<double>() - 2 - exponent);
}

template<typename Int, typename Scalar>
__device__
Scalar
dequantize(const Int &x, const int &e);

template<>
__device__
double
dequantize<long long int, double>(const long long int &x, const int &e)
{
	return LDEXP((double)x, e - (CHAR_BIT * scalar_sizeof<double>() - 2));
}

template<>
__device__
float
dequantize<int, float>(const int &x, const int &e)
{
	return LDEXP((float)x, e - (CHAR_BIT * scalar_sizeof<float>() - 2));
}

template<>
__device__
int
dequantize<int, int>(const int &x, const int &e)
{
	return 1;
}

template<>
__device__
long long int
dequantize<long long int, long long int>(const long long int &x, const int &e)
{
	return 1;
}

/* inverse lifting transform of 4-vector */
template<class Int, uint s>
__host__ __device__
static void
inv_lift(Int* p)
{
	Int x, y, z, w;
	x = *p; p += s;
	y = *p; p += s;
	z = *p; p += s;
	w = *p; p += s;

	/*
	** non-orthogonal transform
	**       ( 4  6 -4 -1) (x)
	** 1/4 * ( 4  2  4  5) (y)
	**       ( 4 -2  4 -5) (z)
	**       ( 4 -6 -4  1) (w)
	*/
	y += w >> 1; w -= y >> 1;
	y += w; w <<= 1; w -= y;
	z += x; x <<= 1; x -= z;
	y += z; z <<= 1; z -= y;
	w += x; x <<= 1; x -= w;

	p -= s; *p = w;
	p -= s; *p = z;
	p -= s; *p = y;
	p -= s; *p = x;
}

/* transform along z */
template<class Int>
 __device__
static void
inv_xform_yx(Int* p)
{
	inv_lift<Int, 16>(p + 1 * threadIdx.x + 4 * threadIdx.z);
}

/* transform along y */
template<class Int>
 __device__
static void
inv_xform_xz(Int* p)
{
	inv_lift<Int, 4>(p + 16 * threadIdx.z + 1 * threadIdx.x);
}

/* transform along x */
template<class Int>
 __device__
static void
inv_xform_zy(Int* p)
{
	inv_lift<Int, 1>(p + 4 * threadIdx.x + 16 * threadIdx.z);
}

/* inverse decorrelating 3D transform */
template<class Int>
 __device__
static void
inv_xform(Int* p)
{

	inv_xform_yx(p);
	__syncthreads();
	inv_xform_xz(p);
	__syncthreads();
	inv_xform_zy(p);
	__syncthreads();
}

/* map two's complement signed integer to negabinary unsigned integer */
inline __host__ __device__
long long int uint2int(unsigned long long int x)
{
	return (x ^0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull;
}

inline __host__ __device__
int uint2int(unsigned int x)
{
	return (x ^0xaaaaaaaau) - 0xaaaaaaaau;
}
template<typename T>
__device__ void print_bits(const T &bits)
{
  const int bit_size = sizeof(T) * 8;

  for(int i = bit_size - 1; i >= 0; --i)
  {
    T one = 1;
    T mask = one << i;
    T val = (bits & mask) >> i ;
    printf("%d", (int) val);
  }
  printf("\n");
}

struct BitStream32
{
  int current_bits; 
  unsigned int bits;
  __device__ BitStream32()
    : current_bits(0), bits(0)
  {
  }

  inline __device__ 
  unsigned short write_bits(const unsigned int &src, 
                            const uint &n_bits)
  {
    // set the first n bits to 0
    unsigned int left = (src >> n_bits) << n_bits;
    unsigned int b = src - left;
    b = b << current_bits;  
    current_bits += n_bits;
    bits += b;
    unsigned int res = left >> n_bits;
    return res;
  }

  inline __device__ 
  unsigned short write_bit(const unsigned short bit)
  {
    bits += bit << current_bits;   
    current_bits += 1;
    return bit;
  }

};

template<int block_size>
struct BlockWriter
{
  int m_word_index;
  int m_start_bit;
  const int m_maxbits; 
  Word *m_words;
  bool m_valid_block;
  __device__ BlockWriter(Word *b, const int &maxbits, const int &block_idx, const int &num_blocks)
    : m_words(b), m_maxbits(maxbits), m_valid_block(true)
  {
    if(block_idx >= num_blocks) m_valid_block = false;
    m_word_index = (block_idx * maxbits)  / (sizeof(Word) * 8); 
    m_start_bit = (block_idx * maxbits) % (sizeof(Word) * 8); 
    //if(m_valid_block)printf("Writer blk_idx %d bsize %d blk_size %d w index %d startbit %d\n", block_idx, bsize, block_size, m_word_index,m_start_bit);
    //printf("***** Writer blk_idx %d bsize %d blk_size %d w index %d startbit %d\n", block_idx, bsize, block_size, m_word_index,m_start_bit);
  }

  inline __device__ 
  void write_bits(const unsigned int &bits, const uint &n_bits, const uint &bit_offset)
  {
    //bool print = m_word_index == 0  && m_start_bit == 0;
    const uint wbits = sizeof(Word) * 8;
    //if(bits == 0) { printf("no\n"); return;}
    uint seg_start = (m_start_bit + bit_offset) % wbits;
    int write_index = m_word_index + (m_start_bit + bit_offset) / wbits;
    uint seg_end = seg_start + n_bits - 1;
    //int write_index = m_word_index;
    uint shift = seg_start; 
    //if(print)
    //{
    //  printf("write index %d shift %d bits %d n_bits %d\n", write_index, seg_start,(int) bits, (int) n_bits);
    //  print_bits(bits);
    //  printf("\n");
    //}
    // handle the case where all of the bits reside in the
    // next word. This is mutually exclusive with straddle.
    //if(seg_start >= sizeof(Word) * 8) 
    //{ 
    //  write_index++;
    //  shift -= sizeof(Word) * 8;
    //}
    // we may be asked to write less bits than exist in 'bits'
    // so we have to make sure that anything after n is zero.
    // If this does not happen, then we may write into a zfp
    // block not at the specified index
    // uint zero_shift = sizeof(Word) * 8 - n_bits;
    Word left = (bits >> n_bits) << n_bits;
    
    Word b = bits - left;
    Word add = b << shift;
    if(m_valid_block) atomicAdd(&m_words[write_index], add); 
    //if(m_valid_block) print_bits(m_words[write_index]);
    // n_bits straddles the word boundary
    bool straddle = seg_start < sizeof(Word) * 8 && seg_end >= sizeof(Word) * 8;
    if(straddle)
    {
      Word rem = b >> (sizeof(Word) * 8 - shift);
      if(m_valid_block) atomicAdd(&m_words[write_index + 1], rem); 
    //  printf("Straddle "); print_bits(rem);
    }
  }

  private:
  __device__ BlockWriter()
  {
  }

};

template<int block_size>
struct BlockReader
{
  int m_current_bit;
  const int m_maxbits; 
  Word *m_words;
 // Word *m_end;
  Word m_buffer;
  bool m_valid_block;
  int m_block_idx;
  __device__ BlockReader(Word *b, const int &maxbits, const int &block_idx, const int &num_blocks)
    :  m_maxbits(maxbits), m_valid_block(true)
  {
    if(block_idx >= num_blocks) m_valid_block = false;
    //if(!m_valid_block) printf("invalid ");
    int word_index = (block_idx * maxbits)  / (sizeof(Word) * 8); 
    int last_index = ((num_blocks - 1) * maxbits)  / (sizeof(Word) * 8); 
    //m_end = (Word *)(b + last_index);
    m_words = b + word_index;
    m_buffer = *m_words;
    m_current_bit = (block_idx * maxbits) % (sizeof(Word) * 8); 
    m_buffer >>= m_current_bit;
    m_block_idx = block_idx;
    //if(m_block_idx == 229) printf("thread %d block id %d word_index %d current bit %d\n", threadIdx.x, block_idx, word_index, m_current_bit);
    //if(m_block_idx == 229) printf("maxbits %d first %d last %d, num_blocks %d\n", maxbits, word_index, last_index, num_blocks);
    //if(block_idx ==0 ) print_bits(m_buffer);
  }

  inline __device__ 
  uint read_bit()
  {
    uint bit = m_buffer & 1;
    ++m_current_bit;
    m_buffer >>= 1;
    // handle moving into next word
    if(m_current_bit >= sizeof(Word) * 8) 
    {
      m_current_bit = 0;
      //if(m_end != m_words) ++m_words;
      ++m_words;
      m_buffer = *m_words;
    }
    return bit; 
  }


  // note this assumes that n_bits is <= 64
  inline __device__ 
  uint read_bits(const int &n_bits)
  {
    uint bits; 
    // rem bits will always be positive
    int rem_bits = sizeof(Word) * 8 - m_current_bit;
     
    int first_read = min(rem_bits, n_bits);
    // first mask 
    Word mask = ((Word)1<<((first_read)))-1;
    bits = m_buffer & mask;
    m_buffer >>= n_bits;
    m_current_bit += first_read;
    //if(m_block_idx == 56) printf(" bits %d rem bits %d n_bits %d first read %d\n", bits, rem_bits, n_bits, first_read);
    int next_read = 0;
    if(n_bits >= rem_bits) 
    {
      //if(m_end != m_words) ++m_words;
      ++m_words;
      m_buffer = *m_words;
      m_current_bit = 0;
      next_read = n_bits - first_read; 
    }
   
    // this is basically a no-op when first read constained 
    // all the bits. TODO: if we have aligned reads, this could 
    // be a conditional without divergence
    mask = ((Word)1<<((next_read)))-1;
    bits += (m_buffer & mask) << first_read;
    m_buffer >>= next_read;
    m_current_bit += next_read; 
    return bits;
  }

  private:
  __device__ BlockReader()
  {
  }

}; // block reader

template<typename Scalar, int Size, typename UInt>
inline __device__
void decode_ints(BlockReader<Size> &reader, uint &max_bits, UInt *data)
{
  const int intprec = get_precision<Scalar>();
  memset(data, 0, sizeof(UInt) * Size);
  unsigned int x; 
  // maxprec = 64;
  const uint kmin = 0; //= intprec > maxprec ? intprec - maxprec : 0;
  int bits = max_bits;
  for (uint k = intprec, n = 0; bits && k-- > kmin;)
  {
    // read bit plane
    uint m = MIN(n, bits);
    bits -= m;
    x = reader.read_bits(m);
    for (; n < Size && bits && (bits--, reader.read_bit()); x += (Word) 1 << n++)
      for (; n < (Size - 1) && bits && (bits--, !reader.read_bit()); n++);
    
    // deposit bit plane
    #pragma unroll
    for (int i = 0; x; i++, x >>= 1)
    {
      data[i] += (UInt)(x & 1u) << k;
    }
  } 
}

template<typename Scalar, int Size, typename UInt>
inline __device__
void decode_ints_debug(BlockReader<Size> &reader, uint &max_bits, UInt *data, int id)
{
  const int intprec = get_precision<Scalar>();
  memset(data, 0, sizeof(UInt) * Size);
  unsigned int x; 
  // maxprec = 64;
  const uint kmin = 0; //= intprec > maxprec ? intprec - maxprec : 0;
  int bits = max_bits;
  for (uint k = intprec, n = 0; bits && k-- > kmin;)
  {
    if(id == 56) {printf("plane %d : ", k); print_bits(reader.m_buffer);}
    //if(id == 56 && k < 4) {printf("next buffer plane %d : ", k); print_bits(reader.m_buffer+1);}
    // read bit plane
    uint m = MIN(n, bits);
    bits -= m;
    x = reader.read_bits(m);
    for (; n < Size && bits && (bits--, reader.read_bit()); x += (Word) 1 << n++)
      for (; n < (Size - 1) && bits && (bits--, !reader.read_bit()); n++);
    
    //printf("x = %d\n", (int) x);
    // deposit bit plane
    #pragma unroll
    for (int i = 0; x; i++, x >>= 1)
    {
      data[i] += (UInt)(x & 1u) << k;
    }
    //printf("bits %d\n", bits);
  } 
}

} // namespace cuZFP
#endif
