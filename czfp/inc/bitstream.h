#ifndef BITSTREAM_H
#define BITSTREAM_H

#include <stddef.h>
#include "types.h"

typedef struct BitStream BitStream;

#ifndef _inline

#ifdef __cplusplus
extern "C"
#endif
/* allocate and initialize bit stream */
BitStream* stream_create(size_t bytes);

/* close and deallocate bit stream */
void stream_close(BitStream* stream);

/* byte size of stream */
size_t stream_size(BitStream* stream);

/* read single bit (0 or 1) */
uint stream_read_bit(BitStream* stream);

/* write single bit */
void stream_write_bit(BitStream* stream, uint bit);

/* read 0 <= n <= 64 bits */
uint64 stream_read_bits(BitStream* stream, uint n);

/* write 0 <= n <= 64 least significant bits of value and return remaining bits */
uint64 stream_write_bits(BitStream* stream, uint64 value, uint n);

/* rewind stream to beginning */
#ifdef __cplusplus
extern "C"
#endif

void stream_rewind(BitStream* stream);

#ifdef __cplusplus
extern "C"
#endif
/* flush out any remaining buffered bits */
void stream_flush(BitStream* stream);

#endif

#endif
