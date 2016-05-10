#ifndef SHARED_H
#define SHARED_H

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

typedef unsigned long long Word;

static const uint wsize = bitsize(Word);
const int intprec = 64;

__constant__ unsigned char c_perm[64];
__constant__ uint c_maxbits;
__constant__ uint c_sizeof_scalar;
__constant__ uint c_maxprec;
__constant__ int c_minexp;
__constant__ int c_ebits;

#endif
