#ifndef ZFP_H
#define ZFP_H

#include "types.h"
#include "system.h"
#include "bitstream.h"

#define ZFP_VERSION 0x0040 /* library version number: 0.4.0 */

/* compression parameters */
typedef struct {
  uint minbits; /* minimum number of bits to store per block */
  uint maxbits; /* maximum number of bits to store per block */
  uint maxprec; /* maximum number of bit planes to store */
  int minexp;   /* minimum bit plane number to store (for floating point) */
} zfp_params;

/* decoder ----------------------------------------------------------------- */

extern void
zfp_decode_block_int32_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int32* iblock);

extern void
zfp_decode_block_int64_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int64* iblock);

extern void
zfp_decode_block_float_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, float* fblock);

extern void
zfp_decode_block_double_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, double* fblock);

extern void
zfp_decode_block_strided_float_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, float* p, uint sx);

extern void
zfp_decode_block_strided_double_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, double* p, uint sx);

extern void
zfp_decode_block_int32_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int32* iblock);

extern void
zfp_decode_block_int64_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int64* iblock);

extern void
zfp_decode_block_float_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, float* fblock);

extern void
zfp_decode_block_double_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, double* fblock);

extern void
zfp_decode_block_strided_float_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, float* p, uint sx, uint sy);

extern void
zfp_decode_block_strided_double_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, double* p, uint sx, uint sy);

extern void
zfp_decode_block_int32_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int32* iblock);

extern void
zfp_decode_block_int64_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int64* iblock);

extern void
zfp_decode_block_float_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, float* fblock);

extern void
zfp_decode_block_double_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, double* fblock);

extern void
zfp_decode_block_strided_float_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, float* p, uint sx, uint sy, uint sz);

extern void
zfp_decode_block_strided_double_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, double* p, uint sx, uint sy, uint sz);

/* encoder ----------------------------------------------------------------- */

extern void
zfp_encode_block_int32_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int32* iblock);

extern void
zfp_encode_block_int64_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int64* iblock);

extern void
zfp_encode_block_float_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const float* fblock);

extern void
zfp_encode_block_double_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const double* fblock);

extern void
zfp_encode_block_strided_float_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const float* p, uint sx);

extern void
zfp_encode_block_strided_double_1(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const double* p, uint sx);

extern void
zfp_encode_block_int32_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int32* iblock);

extern void
zfp_encode_block_int64_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int64* iblock);

extern void
zfp_encode_block_float_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const float* fblock);

extern void
zfp_encode_block_double_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const double* fblock);

extern void
zfp_encode_block_strided_float_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const float* p, uint sx, uint sy);

extern void
zfp_encode_block_strided_double_2(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const double* p, uint sx, uint sy);

extern void
zfp_encode_block_int32_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int32* iblock);

extern void
zfp_encode_block_int64_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int64* iblock);

extern void
zfp_encode_block_float_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const float* fblock);


extern 
#if __cplusplus
"C"
#endif
void
zfp_encode_block_double_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const double* fblock);

extern void
zfp_encode_block_strided_float_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const float* p, uint sx, uint sy, uint sz);

extern void
zfp_encode_block_strided_double_3(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const double* p, uint sx, uint sy, uint sz);

#endif
