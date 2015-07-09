#define EBITS 8
#define NBMASK 0xaaaaaaaau
#if __STDC_VERSION__ >= 199901L
  #define FABS(x)     fabsf(x)
  #define FREXP(x, e) frexpf(x, e)
  #define LDEXP(x, e) ldexpf(x, e)
#else
  #define FABS(x)     (float)fabs(x)
  #define FREXP(x, e) (void)frexp(x, e)
  #define LDEXP(x, e) (float)ldexp(x, e)
#endif
#define Scalar float
#define Int int32
#define UInt uint32

#define EBIAS ((1 << (EBITS - 1)) - 1)
