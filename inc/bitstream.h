#ifndef BITSTREAM_H
#define BITSTREAM_H

#include <stddef.h>
#include "types.h"

typedef struct BitStream BitStream;

// allocate and initialize bit stream
BitStream* stream_create(size_t bytes);

// byte size of stream
size_t stream_size(BitStream* stream);

// write single bit
void stream_write_bit(BitStream* stream, uint bit);

// write n least significant bits of value and return remaining bits
uint64 stream_write_bits(BitStream* stream, uint64 value, uint n);

// flush out any remaining buffered bits
void stream_flush(BitStream* stream);

void stream_seek(BitStream *stream, size_t offset);

#endif
