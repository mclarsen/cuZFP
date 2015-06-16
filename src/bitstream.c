#include <limits.h>
#include <stdlib.h>
#include "bitstream.h"

#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

typedef uint64 Word;

static const uint wsize = bitsize(Word);

// bit stream structure (opaque to caller)
struct BitStream {
  uint bits;   // number of buffered bits (0 <= bits < wsize)
  Word buffer; // buffer for incoming/outgoing bits
  Word* ptr;   // pointer to next word to be read/written
  Word* begin; // beginning of stream
  Word* end;   // end of stream
};

//void
//stream_open(BitStream *stream, void* p, size_t n)
//{
//  stream->begin = stream->ptr = (uchar*)p;
//  stream->end = stream->ptr + n;
//  buffer = 1u;
//}

// allocate and initialize bit stream
BitStream*
stream_create(size_t bytes)
{
  BitStream* stream = malloc(sizeof(BitStream));
  stream->bits = 0;
  stream->buffer = 0;
  stream->ptr = malloc(bytes);
  stream->begin = stream->ptr;
  stream->end = stream->begin + bytes / sizeof(Word);
  return stream;
}

// byte size of stream
size_t
stream_size(BitStream* stream)
{
  return sizeof(Word) * (stream->ptr - stream->begin);
}

// write single bit (must be 0 or 1)
void
stream_write_bit(BitStream* stream, uint bit)
{
  stream->buffer += (Word)bit << stream->bits;
  if (++stream->bits == wsize) {
    *stream->ptr++ = stream->buffer;
    stream->buffer = 0;
    stream->bits = 0;
  }
}

// write 0 <= n <= 64 least significant bits of value and return remaining bits
uint64
stream_write_bits(BitStream* stream, uint64 value, uint n)
{
  if (n == bitsize(value)) {
    if (!stream->bits)
      *stream->ptr++ = value;
    else {
      *stream->ptr++ = stream->buffer + (value << stream->bits);
      stream->buffer = value >> (bitsize(value) - stream->bits);
    }
    return 0;
  }
  else {
    uint64 v = value >> n;
    value -= v << n;
    stream->buffer += value << stream->bits;
    stream->bits += n;
    if (stream->bits >= wsize) {
      stream->bits -= wsize;
      *stream->ptr++ = stream->buffer;
      stream->buffer = value >> (n - stream->bits);
    }
    return v;
  }
}

// flush out any remaining buffered bits
void
stream_flush(BitStream* stream)
{
  if (stream->bits)
    stream_write_bits(stream, 0, wsize - stream->bits);
}

void
stream_seek(BitStream *stream, size_t offset)
{
  stream->ptr = stream->begin + offset/wsize;
  stream->bits = 0;
  stream->buffer = 0u;
}

