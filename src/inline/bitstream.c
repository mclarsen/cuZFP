#include <limits.h>
#include <stdlib.h>

#ifndef _inline
  #define _inline
#endif

#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

typedef uint64 Word; /* bit buffer type */

enum {
  wsize = bitsize(Word) /* number of bits in a buffered word */
};

/* bit stream structure (opaque to caller) */
struct BitStream {
  uint bits;   /* number of buffered bits (0 <= bits < wsize) */
  Word buffer; /* buffer for incoming/outgoing bits */
  Word* ptr;   /* pointer to next word to be read/written */
  Word* begin; /* beginning of stream */
  Word* end;   /* end of stream (unused) */
};

/* byte size of stream (if flushed) */
_inline size_t
stream_size(BitStream* stream)
{
  return sizeof(Word) * (stream->ptr - stream->begin);
}

/* read single bit (0 or 1) */
_inline uint
stream_read_bit(BitStream* stream)
{
  uint bit;
  if (!stream->bits) {
    stream->buffer = *stream->ptr++;
    stream->bits = wsize;
  }
  stream->bits--;
  bit = (uint)stream->buffer & 1u;
  stream->buffer >>= 1;
  return bit;
}

/* write single bit (must be 0 or 1) */
_inline void
stream_write_bit(BitStream* stream, uint bit)
{
  stream->buffer += (Word)bit << stream->bits;
  if (++stream->bits == wsize) {
    *stream->ptr++ = stream->buffer;
    stream->buffer = 0;
    stream->bits = 0;
  }
}

/* read 0 <= n <= 64 bits */
_inline uint64
stream_read_bits(BitStream* stream, uint n)
{
#if 0
  /* read bits in LSB to MSB order */
  uint64 value = 0;
  for (uint i = 0; i < n; i++)
    value += (uint64)stream_read_bit(stream) << i;
  return value;
#elif 1
  uint64 value;
  /* because shifts by 64 are not possible, treat n = 64 specially */
  if (n == bitsize(value)) {
    if (!stream->bits)
      value = *stream->ptr++;
    else {
      value = stream->buffer;
      stream->buffer = *stream->ptr++;
      value += stream->buffer << stream->bits;
      stream->buffer >>= n - stream->bits;
    }
  }
  else {
    value = stream->buffer;
    if (stream->bits < n) {
      /* not enough bits buffered; fetch wsize more */
      stream->buffer = *stream->ptr++;
      value += stream->buffer << stream->bits;
      stream->buffer >>= n - stream->bits;
      stream->bits += wsize;
    }
    else
      stream->buffer >>= n;
    value -= stream->buffer << n;
    stream->bits -= n;
  }
  return value;
#endif
}

/* write 0 <= n <= 64 least significant bits of value and return remaining bits */
_inline uint64
stream_write_bits(BitStream* stream, uint64 value, uint n)
{
  /* because shifts by 64 are not possible, treat n = 64 specially */
  if (n == bitsize(value)) {
    /* output all 64 bits */
    if (!stream->bits)
      *stream->ptr++ = value;
    else {
      *stream->ptr++ = stream->buffer + (value << stream->bits);
      stream->buffer = value >> (bitsize(value) - stream->bits);
    }
    return 0;
  }
  else {
    /* split value into bits to output and remaining bits */
    uint64 v = value >> n;
    value -= v << n;
    /* append bit string to buffer */
    stream->buffer += value << stream->bits;
    stream->bits += n;
    /* is buffer full? */
    if (stream->bits >= wsize) {
      /* outbut buffer */
      stream->bits -= wsize;
      *stream->ptr++ = stream->buffer;
      stream->buffer = value >> (n - stream->bits);
    }
    return v;
  }
}

/* rewind stream to beginning */
_inline void
stream_rewind(BitStream* stream)
{
  stream->ptr = stream->begin;
  stream->bits = 0;
  stream->buffer = 0;
}

/* flush out any remaining buffered bits */
_inline void
stream_flush(BitStream* stream)
{
  if (stream->bits)
    stream_write_bits(stream, 0, wsize - stream->bits);
}

/* allocate and initialize bit stream */
_inline BitStream*
stream_create(size_t bytes)
{
  BitStream* stream = malloc(sizeof(BitStream));
  if (stream) {
    stream->begin = malloc(bytes);
    stream->end = stream->begin + bytes / sizeof(Word);
    stream_rewind(stream);
  }
  return stream;
}

/* close and deallocate bit stream */
_inline void
stream_close(BitStream* stream)
{
  free(stream->begin);
  free(stream);
}

#undef bitsize
