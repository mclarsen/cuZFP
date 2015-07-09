#define EBITS 11
#define NBMASK 0xaaaaaaaaaaaaaaaaull
#define FABS(x) fabs(x)
#define FREXP(x, e) frexp(x, e)
#define LDEXP(x, e) ldexp(x, e)
#define Scalar double
#define Int int64
#define UInt uint64

#define EBIAS ((1 << (EBITS - 1)) - 1)
