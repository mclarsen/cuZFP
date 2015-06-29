#include <limits.h>
#include <stdlib.h>

#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

typedef unsigned long long Word;

static const uint wsize = bitsize(Word);

// bit stream structure (opaque to caller)
class BitStream {
public:
  uint bits;   // number of buffered bits (0 <= bits < wsize)
  Word buffer; // buffer for incoming/outgoing bits
  Word* ptr;   // pointer to next word to be read/written
  Word* begin; // beginning of stream
  Word* end;   // end of stream

  BitStream(){}
  // byte size of stream
  __device__ __host__
  size_t
  size()
  {
    return sizeof(Word) * (ptr - begin);
  }

  // write single bit (must be 0 or 1)
  __device__ __host__
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
  __device__ __host__
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
  seek( size_t offset)
  {
    ptr = begin + offset/wsize;
    bits = 0;
    buffer = 0u;
  }


};

template<class UInt>
__device__ __host__
static void
encode_ints(BitStream* stream, const UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long count, uint size)
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

// allocate and initialize bit stream
BitStream*
stream_create(size_t bytes)
{
  BitStream* stream = new BitStream();//malloc(sizeof(BitStream));
  stream->bits = 0;
  stream->buffer = 0;
  stream->ptr = new Word[bytes];
  stream->begin = stream->ptr;
  stream->end = stream->begin + bytes / sizeof(Word);
  return stream;
}

