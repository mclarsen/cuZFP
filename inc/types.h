#ifndef TYPES_H
#define TYPES_H

typedef unsigned char uchar;
typedef unsigned int uint;
<<<<<<< HEAD
typedef signed int int32;
typedef unsigned int uint32;
typedef signed long long int64;
typedef unsigned long long uint64;

=======

#if __STDC_VERSION__ >= 199901L
  #include <stdint.h>
  typedef int32_t int32;
  typedef uint32_t uint32;
  typedef int64_t int64;
  typedef uint64_t uint64;
#else
  typedef signed int int32;
  typedef unsigned int uint32;
  typedef signed long long int64; /* not ANSI C compliant */
  typedef unsigned long long uint64; /* not ANSI C compliant */
#endif

>>>>>>> master
#endif
