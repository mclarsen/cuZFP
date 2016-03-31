static void _t1(inv_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* scatter 4*4*4 block to strided array */
static void
_t2(scatter, Scalar, 3)(const Scalar* q, Scalar* p, uint sx, uint sy, uint sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++, p += sz - 4 * sy)
    for (y = 0; y < 4; y++, p += sy - 4 * sx)
      for (x = 0; x < 4; x++, p += sx)
        *p = *q++;
}

/* inverse decorrelating 3D transform */
static void
_t2(inv_xform, Int, 3)(Int* p)
{
  uint x, y, z;
  /* transform along z */
  for (y = 0; y < 4; y++)
    for (x = 0; x < 4; x++)
      _t1(inv_lift, Int)(p + 1 * x + 4 * y, 16);
  /* transform along y */
  for (x = 0; x < 4; x++)
    for (z = 0; z < 4; z++)
      _t1(inv_lift, Int)(p + 16 * z + 1 * x, 4);
  /* transform along x */
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      _t1(inv_lift, Int)(p + 4 * y + 16 * z, 1);
}

/* public functions -------------------------------------------------------- */

/* decode 4*4*4 floating-point block and store at p using strides (sx, sy, sz) */
void
_t2(zfp_decode_block_strided, Scalar, 3)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, Scalar* p, uint sx, uint sy, uint sz)
{
  /* decode contiguous block */
  Scalar fblock[64];
  _t2(zfp_decode_block, Scalar, 3)(stream, minbits, maxbits, maxprec, minexp, fblock);
  /* scatter block to strided array */
  _t2(scatter, Scalar, 3)(fblock, p, sx, sy, sz);
}
