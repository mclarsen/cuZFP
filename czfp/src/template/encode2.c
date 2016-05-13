static void _t1(fwd_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* gather 4*4 block from strided array */
static void
_t2(gather, Scalar, 2)(Scalar* q, const Scalar* p, uint sx, uint sy)
{
  uint x, y;
  for (y = 0; y < 4; y++, p += sy - 4 * sx)
    for (x = 0; x < 4; x++, p += sx)
      *q++ = *p;
}

/* forward decorrelating 2D transform */
static void
_t2(fwd_xform, Int, 2)(Int* p)
{
  uint x, y;
  /* transform along x */
  for (y = 0; y < 4; y++)
    _t1(fwd_lift, Int)(p + 4 * y, 1);
  /* transform along y */
  for (x = 0; x < 4; x++)
    _t1(fwd_lift, Int)(p + 1 * x, 4);
}

/* public functions -------------------------------------------------------- */

/* encode 4*4 floating-point block stored at p using strides (sx, sy) */
void
_t2(zfp_encode_block_strided, Scalar, 2)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const Scalar* p, uint sx, uint sy)
{
  /* gather block from strided array */
  Scalar fblock[16];
  _t2(gather, Scalar, 2)(fblock, p, sx, sy);
  /* encode floating-point block */
  _t2(zfp_encode_block, Scalar, 2)(stream, minbits, maxbits, maxprec, minexp, fblock);
}
