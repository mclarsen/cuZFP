#ifndef BITSTREAM_CUH
#define BITSTREAM_CUH

#include <limits.h>
#include <stdlib.h>
#include "shared.h"

namespace cuZFP{

template<uint bsize>
class Bit
{
public:
	int emax;
    uint bits;   // number of buffered bits (0 <= bits < wsize)
    Word buffer; // buffer for incoming/outgoing bits
    char offset;   // pointer to next word to be read/written
    Word begin[bsize]; // beginning of stream
    Word* end;   // end of stream
    unsigned long long x;

    __device__ __host__
    Bit(){
			emax = 0;
        bits = 0;
        buffer = 0;
        offset = 0;
        for (int i=0; i<bsize; i++){
            begin[i] = 0;
        }

        end = begin + bsize;


        x = 0;
    }

    __device__ __host__
    Bit(const Bit &bs)
    {
			emax = bs.emax;
        bits = bs.bits;
        buffer = bs.buffer;
        offset = bs.offset;
        for (int i=0; i<bsize; i++){
            begin[i] = bs.begin[i];
        }
        end = begin + bsize;

        x = bs.x;
    }

    // byte size of stream
    __device__ __host__
    size_t
    size()
    {
      return sizeof(Word) * (offset);
    }

    // write single bit (must be 0 or 1)
    __device__ __host__
    uint
    write_bit(uint bit)
    {
      buffer += (Word)bit << bits;
      if (++bits == wsize) {
        begin[offset++] = buffer;
        buffer = 0;
        bits = 0;
      }
			return bit;
    }


    // write 0 <= n <= 64 least significant bits of value and return remaining bits
    __device__ __host__
    unsigned long long
    write_bits(unsigned long long value, uint n)
    {
      if (n == bitsize(value)) {
        if (!bits)
          begin[offset++] = value;
        else {
          begin[offset++] = buffer + (value << bits);
          buffer = value >> (bitsize(value) - bits);
        }
        return 0;
      }
      else {
        unsigned long long v = value >> n;
        value -= v << n;
        buffer += value << bits;
        bits += n;
        if (bits >= wsize) {
          bits -= wsize;
          begin[offset++] = buffer;
          buffer = value >> (n - bits);
        }
        return v;
      }
    }

    // flush out any remaining buffered bits
    __device__ __host__
    void
    flush()
    {
      if (bits)
        write_bits( 0, wsize - bits);
    }

    __device__ __host__
    void
    seek( size_t _offset)
    {
      offset = _offset/wsize;
      bits = 0;
      buffer = 0u;
    }


    __device__ __host__
    void
    seek( char _offset, uint _bits, Word _buffer)
    {
      offset = _offset;
      bits = _bits;
      buffer = _buffer;
    }
    /*****************************************************************************************/
    /* read single bit (0 or 1) */
    __host__ __device__
    int
    read_bit()
    {
      uint bit;
      if (!bits) {
        buffer = begin[offset++];
        bits = wsize;
      }
      bits--;
      bit = (uint)buffer & 1u;
      buffer >>= 1;
      return bit;
    }
    /* read 0 <= n <= 64 bits */
    __host__ __device__
    unsigned long long
    read_bits(uint n)
    {
    #if 0
      /* read bits in LSB to MSB order */
      uint64 value = 0;
      for (uint i = 0; i < n; i++)
        value += (uint64)stream_read_bit(stream) << i;
      return value;
    #elif 1
      unsigned long long value;
      /* because shifts by 64 are not possible, treat n = 64 specially */
      if (n == bitsize(value)) {
        if (!bits)
          value = begin[offset++];//*ptr++;
        else {
          value = buffer;
          buffer = begin[offset++];//*ptr++;
          value += buffer << bits;
          buffer >>= n - bits;
        }
      }
      else {
        value = buffer;
        if (bits < n) {
          /* not enough bits buffered; fetch wsize more */
          buffer = begin[offset++];//*ptr++;
          value += buffer << bits;
          buffer >>= n - bits;
          bits += wsize;
        }
        else
          buffer >>= n;
        value -= buffer << n;
        bits -= n;
      }
      return value;
    #endif
    }
    /* rewind stream to beginning */
    __host__ __device__
    void
    rewind()
    {
      offset = 0;//ptr = begin;
      bits = 0;
      buffer = 0;
    }
};

// bit stream structure (opaque to caller)
class BitStream {
public:
  uint bits;   // number of buffered bits (0 <= bits < wsize)
  Word buffer; // buffer for incoming/outgoing bits
  Word* ptr;   // pointer to next word to be read/written
  Word* begin; // beginning of stream
  Word* end;   // end of stream

  // byte size of stream
   __host__
  size_t
  size()
  {
    return sizeof(Word) * (ptr - begin);
  }

  // write single bit (must be 0 or 1)
  __host__ __device__
  void
  write_bit(uint bit)
  {
    buffer += (Word)bit << bits;
    if (++bits == wsize) {
      *ptr++ = buffer;
      buffer = 0;
      bits = 0;
    }
  }


  // write 0 <= n <= 64 least significant bits of value and return remaining bits
  __host__ __device__
  unsigned long long
  write_bits(unsigned long long value, uint n)
  {
    if (n == bitsize(value)) {
      if (!bits)
        *ptr++ = value;
      else {
        *ptr++ = buffer + (value << bits);
        buffer = value >> (bitsize(value) - bits);
      }
      return 0;
    }
    else {
      unsigned long long v = value >> n;
      value -= v << n;
      buffer += value << bits;
      bits += n;
      if (bits >= wsize) {
        bits -= wsize;
        *ptr++ = buffer;
        buffer = value >> (n - bits);
      }
      return v;
    }
  }


  /*****************************************************************************************/
  /* read single bit (0 or 1) */
  __host__
  int
  read_bit()
  {
    uint bit;
    if (!bits) {
      buffer = *ptr++;
      bits = wsize;
    }
    bits--;
    bit = (uint)buffer & 1u;
    buffer >>= 1;
    return bit;
  }
  /* read 0 <= n <= 64 bits */
  __host__
  unsigned long long
  read_bits(uint n)
  {
  #if 0
    /* read bits in LSB to MSB order */
    uint64 value = 0;
    for (uint i = 0; i < n; i++)
      value += (uint64)stream_read_bit(stream) << i;
    return value;
  #elif 1
    unsigned long long value;
    /* because shifts by 64 are not possible, treat n = 64 specially */
    if (n == bitsize(value)) {
      if (!bits)
        value = *ptr++;
      else {
        value = buffer;
        buffer = *ptr++;
        value += buffer << bits;
        buffer >>= n - bits;
      }
    }
    else {
      value = buffer;
      if (bits < n) {
        /* not enough bits buffered; fetch wsize more */
        buffer = *ptr++;
        value += buffer << bits;
        buffer >>= n - bits;
        bits += wsize;
      }
      else
        buffer >>= n;
      value -= buffer << n;
      bits -= n;
    }
    return value;
  #endif
  }
  // flush out any remaining buffered bits
  __host__
  void
  flush()
  {
    if (bits)
      write_bits( 0, wsize - bits);
  }

  /* rewind stream to beginning */
  __host__
  void
  rewind()
  {
    ptr = begin;
    bits = 0;
    buffer = 0;
  }

  __host__
  void
  seek( size_t offset)
  {
    ptr = begin + offset/wsize;
    bits = 0;
    buffer = 0u;
  }


};

// maximum number of bit planes to encode
__device__ __host__
static uint
precision(int maxexp, uint maxprec, int minexp)
{
  return MIN(maxprec, MAX(0, maxexp - minexp + 8));
}

template<class UInt>
 __host__
static void
encode_ints_old(BitStream* stream, const UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long count, uint size)
{
  if (!maxbits)
    return;

  uint bits = maxbits;
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;

  // output one bit plane at a time from MSB to LSB
  for (uint k = intprec, n = 0; k-- > kmin;) {
    // extract bit plane k to x
    unsigned long long x = 0;
    for (uint i = 0; i < size; i++)
      x += ((data[i] >> k) & (unsigned long long)1) << i;
    // encode bit plane
    for (uint m = n;; m = count & 0xfu, count >>= 4, n += m) {
      // encode bit k for next set of m values
      m = MIN(m, bits);
      bits -= m;
      x = stream->write_bits(x, m);
      // exit if there are no more bits to write
      if (!bits)
        return;
      // continue with next bit plane if out of groups or group test passes
      if (!count || (bits--, stream->write_bit(!!x), !x))
        break;
    }
  }

  // pad with zeros in case fewer than minbits bits have been written
  while (bits-- > maxbits - minbits)
    stream->write_bit(0);
}


/* compress sequence of unsigned integers */
template<class UInt>
 __device__ __host__
void
encode_ints_old_par(BitStream* stream, const UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long count, uint size)
{
    uint intprec = CHAR_BIT * (uint)sizeof(UInt);
    uint kmin = intprec > maxprec ? intprec - maxprec : 0;
    uint bits = maxbits;
    uint i, k, m, n;
    unsigned long long x[CHAR_BIT * sizeof(UInt)], y, c;
    uint g[CHAR_BIT * sizeof(UInt)], h;

    if (!maxbits)
     return;

    /* parallel: extract and group test bit planes */
    for (k = kmin; k < intprec; k++) {
     /* extract bit plane k to x[k] */
     y = 0;
     for (i = 0; i < size; i++)
       y += ((data[i] >> k) & (unsigned long long)1) << i;
     x[k] = y;
     /* count number of positive group tests g[k] among 3*d in d dimensions */
     h = 0;
     for (c = count; y; y >>= c & 0xfu, c >>= 4)
       h++;
     g[k] = h;
    }

    /* serial: output one bit plane at a time from MSB to LSB */
    for (k = intprec, n = 0, h = 0; k-- > kmin;) {
     /* encode bit k for first n values */
     y = x[k];
     if (n < bits) {
       y = stream->write_bits(y, n);
       bits -= n;
     }
     else {
       stream->write_bits(y, bits);
       bits = 0;
       return;
     }
     /* perform series of group tests */
     while (h < g[k]) {
       /* output a one bit for a positive group test */
       stream->write_bit(1);
       bits--;
       /* add next group of m values to significant set */
       m = count & 0xfu;
       count >>= 4;
       n += m;
       /* encode next group of m values */
       if (m < bits) {
         y = stream->write_bits( y, m);
         bits -= m;
       }
       else {
         stream->write_bits(y, bits);
         bits = 0;
         return;
       }
       h++;
     }
     /* if there are more groups, output a zero bit for a negative group test */
     if (count) {
       stream->write_bit(0);
       bits--;
     }
    }

    /* write at least minbits bits by padding with zeros */
    while (bits > maxbits - minbits) {
     stream->write_bit(0);
     bits--;
    }
}

 template<class UInt>
 void
	 __device__ __host__
	 extract_bit
	 (
	 uint k,
	 unsigned long long &x,
	 unsigned char &g,
	 const UInt *data,
	 unsigned long long count,
	 uint size
	 )
 {
	 /* extract bit plane k to x[k] */
	 unsigned long long y = 0;
	 for (uint i = 0; i < size; i++)
		 y += ((data[i] >> k) & (unsigned long long)1) << i;
	 x = y;
	 /* count number of positive group tests g[k] among 3*d in d dimensions */
	 g = 0;
	 for (unsigned long long c = count; y; y >>= c & 0xfu, c >>= 4)
		 g++;

 }


template<class UInt>
void
__device__ __host__
extract_bit
(
        uint k,
        unsigned long long &x,
        uint &g,
        const UInt *data,
        unsigned long long count,
        uint size
     )
{
 /* extract bit plane k to x[k] */
 unsigned long long y = 0;
 for (uint i = 0; i < size; i++)
   y += ((data[i] >> k) & (unsigned long long)1) << i;
 x = y;
 /* count number of positive group tests g[k] among 3*d in d dimensions */
 g = 0;
 for (unsigned long long c = count; y; y >>= c & 0xfu, c >>= 4)
   g++;

}


template<class UInt, uint bsize>
__device__ __host__
void
encode_group_test(unsigned long long *x, uint *g, const UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long count, uint size)
 {
     uint intprec = CHAR_BIT * (uint)sizeof(UInt);
     uint kmin = intprec > maxprec ? intprec - maxprec : 0;


     if (!maxbits)
      return;

     /* parallel: extract and group test bit planes */
//#pragma omp parallel for
     for (int k = kmin; k < intprec; k++) {
        extract_bit(k, x[k], g[k], data, count, size);
     }
}

template<class UInt, uint bsize>
__device__ __host__
void encode_bit_plane(const unsigned long long *x, const uint *g, Bit<bsize> & stream, const UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long count)
{
    uint m, n;
     uint h = 0, k = 0;
     uint kmin = intprec > maxprec ? intprec - maxprec : 0;
     uint bits = maxbits;
     /* serial: output one bit plane at a time from MSB to LSB */
     for (k = intprec, n = 0, h = 0; k-- > kmin;) {
      /* encode bit k for first n values */
      unsigned long long y = x[k];
      if (n < bits) {
        y = stream.write_bits(y, n);
        bits -= n;
      }
      else {
        stream.write_bits(y, bits);
        bits = 0;
        return;
      }
      /* perform series of group tests */
      while (h < g[k]) {
        /* output a one bit for a positive group test */
        stream.write_bit(1);
        bits--;
        /* add next group of m values to significant set */
        m = count & 0xfu;
        count >>= 4;
        n += m;
        /* encode next group of m values */
        if (m < bits) {
          y = stream.write_bits( y, m);
          bits -= m;
        }
        else {
          stream.write_bits(y, bits);
          bits = 0;
          return;
        }
        h++;
      }
      /* if there are more groups, output a zero bit for a negative group test */
      if (count) {
        stream.write_bit(0);
        bits--;
      }
     }

     /* write at least minbits bits by padding with zeros */
     while (bits > maxbits - minbits) {
      stream.write_bit(0);
      bits--;
     }
 }
template<class UInt, uint bsize>
__device__ __host__
static void
encode_ints_par(Bit<bsize> & stream, const UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long count, uint size)
{
    unsigned long long x[CHAR_BIT * sizeof(UInt)];
    uint g[CHAR_BIT * sizeof(UInt)];

    encode_group_test<UInt, bsize>(x, g, data, minbits, maxbits, maxprec, count, size);
    encode_bit_plane<UInt, bsize>(x, g, stream, data, minbits, maxbits, maxprec, count);

}

template<class UInt, uint bsize>
__device__ __host__
void encode_bitplane
(
        Bit<bsize> &stream,
        const UInt *data,
        unsigned long long &count,
        uint size,
        uint &bits,
        uint &n,
        const uint k
        )
{
    // extract bit plane k to x
    stream.x = 0;
    for (uint i = 0; i < size; i++)
      stream.x += ((data[i] >> k) & (unsigned long long)1) << i;
    // encode bit plane
    for (uint m = n;; m = count & 0xfu, count >>= 4, n += m) {
      // encode bit k for next set of m values
      m = MIN(m, bits);
      bits -= m;
      stream.x = stream.write_bits(stream.x, m);
      // continue with next bit plane if out of groups or group test passes
      if (!count || (bits--, stream.write_bit(!!stream.x), !stream.x))
        break;
    }
}
template<class UInt, uint bsize>
__device__ __host__
static void
encode_ints(Bit<bsize> & stream, const UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long count, uint size)
{
  if (!maxbits)
    return;

  uint bits = maxbits;
  uint kmin = intprec > maxprec ? intprec - maxprec : 0;

  // output one bit plane at a time from MSB to LSB
  for (uint k = intprec, n = 0; k-- > kmin;) {
      if (bits){
          encode_bitplane(stream, data, count, size, bits, n, k);
      }
  }

  // pad with zeros in case fewer than minbits bits have been written
  while (bits-- > maxbits - minbits)
    stream.write_bit(0);
}
// allocate and initialize bit stream
BitStream*
stream_create(size_t bytes)
{
  BitStream* stream = new BitStream();//malloc(sizeof(BitStream));
  stream->bits = 0;
  stream->buffer = 0;
  stream->ptr = new Word[bytes];
  stream->begin = stream->ptr;
  stream->end = stream->begin + bytes;
  return stream;
}

template< class UInt, uint bsize>
__global__
void cudaencode
(
        const UInt *q,
        Bit<bsize> *stream,
        const int *emax,
        uint minbits, uint maxbits, uint maxprec, int minexp, unsigned long long group_count, uint size

        )
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    extern __shared__ Bit<bsize> sl_bits[];

		sl_bits[threadIdx.x] = stream[idx];

		encode_ints<UInt, bsize>(sl_bits[threadIdx.x], q + idx * bsize, minbits, maxbits, precision(emax[idx], maxprec, minexp), group_count, size);
		stream[idx] = sl_bits[threadIdx.x];
//    encode_ints<UInt, bsize>(stream[idx], q + idx * bsize, minbits, maxbits, precision(emax[idx], maxprec, minexp), group_count, size);

}
}
#endif
