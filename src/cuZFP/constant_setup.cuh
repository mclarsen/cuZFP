#ifndef cuZFP_CONSTANT_SETUP
#define cuZFP_CONSTANT_SETUP

#include <constants.h>
#include <shared.h>
#include <ErrorCheck.h>
#include <type_info.cuh>

namespace cuZFP {

class ConstantSetup
{
public:
  template<typename Scalar>
  static void setup_3d(Scalar, const int rate)
  { 
    ErrorCheck ec;
    cudaMemcpyToSymbol(c_perm, perm_3d, sizeof(unsigned char) * 64, 0); 
    ec.chk("setupConst: c_perm");
  }
};

} //namespace 

#endif
