#ifndef SHARED_H
#define SHARED_H

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define bitsize(x) (CHAR_BIT * (uint)sizeof(x))

typedef unsigned long long Word;

static const uint wsize = bitsize(Word);
const int intprec = 64;

#endif
