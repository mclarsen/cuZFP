#include <limits.h>
#include <stdlib.h>

#include "bitstream.h"
#include "codec.h"

const uint nx = 32;
const uint ny = 32;
const uint nz = 32;
uint mx = 0;
uint my = 0;
uint mz = 0;
const double rate = 64;
uchar *dims  = 0;
size_t blksize = 0;


static size_t block_size(double rate) { return (lrint(64 * rate) + CHAR_BIT - 1) / CHAR_BIT; }

// initialize array by copying and compressing data stored at p
void set(BitStream *stream, const Scalar* p)
{
  size_t offset = 0;
  for (uint k = 0; k < mz; k++, p += 4 * nx * (ny - my))
    for (uint j = 0; j < my; j++, p += 4 * (nx - mx))
      for (uint i = 0; i < mx; i++, p += 4, offset += blksize) {
        stream_seek(stream, offset);
        encode3(stream, p, 1, nx, nx * ny,0,1013,64,-1074);
        stream_flush(stream);
      }
  //cache.clear();
}

void init(Scalar *p){
    for (int z=0; z<nz; z++){
        for (int y=0; y<ny; y++){
            for (int x=0; x<nx; x++){
                p[z * nx*ny + y*nx + x] = rand()/(double)RAND_MAX;
            }
        }
    }
}

int main()
{
    double *p = malloc(sizeof(double) *nx*ny*nz);
    mx = nx / 4;
    my = ny / 4;
    mz = nz / 4;
    blksize = block_size(rate);
    BitStream *bs = stream_create(blksize*mx*my);

    init(p);
    set(bs, p);

    free(p);
    free(bs);
    return 0;
}
