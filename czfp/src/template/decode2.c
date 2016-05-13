static void _t1(inv_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* scatter 4*4 block to strided array */
static void
_t2(scatter, Scalar, 2)(const Scalar* q, Scalar* p, uint sx, uint sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

/* inverse decorrelating 2D transform */
static void
_t2(inv_xform, Int, 2)(Int* p)
{
  uint x, y;
  /* transform along y */
  for (x = 0; x < 4; x++)
    _t1(inv_lift, Int)(p + 1 * x, 4);
  /* transform along x */
  for (y = 0; y < 4; y++)
    _t1(inv_lift, Int)(p + 4 * y, 1);
}

/* public functions -------------------------------------------------------- */

/* decode 4*4 floating-point block and store at p using strides (sx, sy) */
void
_t2(zfp_decode_block_strided, Scalar, 2)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, Scalar* p, uint sx, uint sy)
{
  /* decode contiguous block */
  Scalar fblock[16];
  _t2(zfp_decode_block, Scalar, 2)(stream, minbits, maxbits, maxprec, minexp, fblock);
  /* scatter block to strided array */
  _t2(scatter, Scalar, 2)(fblock, p, sx, sy);
}
