#ifndef cuZFP_h
#define cuZFP_h

#include <stdio.h>
#include <zfp_structs.h>

namespace cuZFP {

void compress(zfp_stream *stream, zfp_field *field);
void decompress(zfp_stream *stream, zfp_field *field);

} // namespace cuZFP


#endif
