static void _t1(inv_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* scatter 4-value block to strided array */
static void
_t2(scatter, Scalar, 1)(const Scalar* q, Scalar* p, uint sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

/* inverse decorrelating 1D transform */
static void
_t2(inv_xform, Int, 1)(Int* p)
{
  /* transform along x */
  _t1(inv_lift, Int)(p, 1);
}

/* public functions -------------------------------------------------------- */

/* decode 4-value floating-point block and store at p using stride sx */
void
_t2(zfp_decode_block_strided, Scalar, 1)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, Scalar* p, uint sx)
{
  /* decode contiguous block */
  Scalar fblock[4];
  _t2(zfp_decode_block, Scalar, 1)(stream, minbits, maxbits, maxprec, minexp, fblock);
  /* scatter block to strided array */
  _t2(scatter, Scalar, 1)(fblock, p, sx);
}
