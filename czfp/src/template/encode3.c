static void _t1(fwd_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* gather 4*4*4 block from strided array */
static void
_t2(gather, Scalar, 3)(Scalar* q, const Scalar* p, uint sx, uint sy, uint sz)
{
  uint x, y, z;
  for (z = 0; z < 4; z++, p += sz - 4 * sy)
    for (y = 0; y < 4; y++, p += sy - 4 * sx)
      for (x = 0; x < 4; x++, p += sx)
        *q++ = *p;
}

/* forward decorrelating 3D transform */
static void
_t2(fwd_xform, Int, 3)(Int* p)
{
  uint x, y, z;
  /* transform along x */
  for (z = 0; z < 4; z++)
    for (y = 0; y < 4; y++)
      _t1(fwd_lift, Int)(p + 4 * y + 16 * z, 1);
  /* transform along y */
  for (x = 0; x < 4; x++)
    for (z = 0; z < 4; z++)
      _t1(fwd_lift, Int)(p + 16 * z + 1 * x, 4);
  /* transform along z */
  for (y = 0; y < 4; y++)
    for (x = 0; x < 4; x++)
      _t1(fwd_lift, Int)(p + 1 * x + 4 * y, 16);
}

/* public functions -------------------------------------------------------- */

/* encode 4*4*4 floating-point block stored at p using strides (sx, sy, sz) */
void
_t2(zfp_encode_block_strided, Scalar, 3)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const Scalar* p, uint sx, uint sy, uint sz)
{
  /* gather block from strided array */
  Scalar fblock[64];
  _t2(gather, Scalar, 3)(fblock, p, sx, sy, sz);
  /* encode floating-point block */
  _t2(zfp_encode_block, Scalar, 3)(stream, minbits, maxbits, maxprec, minexp, fblock);
}
