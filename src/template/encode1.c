static void _t1(fwd_lift, Int)(Int* p, uint s);

/* private functions ------------------------------------------------------- */

/* gather 4-value block from strided array */
static void
_t2(gather, Scalar, 1)(Scalar* q, const Scalar* p, uint sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *q++ = *p;
}

/* forward decorrelating 1D transform */
static void
_t2(fwd_xform, Int, 1)(Int* p)
{
  /* transform along x */
  _t1(fwd_lift, Int)(p, 1);
}

/* public functions -------------------------------------------------------- */

/* encode 4-value floating-point block stored at p using stride sx */
void
_t2(zfp_encode_block_strided, Scalar, 1)(BitStream* stream, uint minbits, uint maxbits, uint maxprec, int minexp, const Scalar* p, uint sx)
{
  /* gather block from strided array */
  Scalar fblock[4];
  _t2(gather, Scalar, 1)(fblock, p, sx);
  /* encode floating-point block */
  _t2(zfp_encode_block, Scalar, 1)(stream, minbits, maxbits, maxprec, minexp, fblock);
}
