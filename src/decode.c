#include <math.h>
#include "codec.h"

// map negabinary unsigned integer to two's complement signed integer
static Int
uint2int(UInt y)
{
  return (y ^ (UInt)0xaaaaaaaaaaaaaaaaull) - (UInt)0xaaaaaaaaaaaaaaaaull;
}
