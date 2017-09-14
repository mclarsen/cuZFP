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
  static void setup_3d()
  { 
    ErrorCheck ec;
    cudaMemcpyToSymbol(c_perm, perm_3d, sizeof(unsigned char) * 64, 0); 
    ec.chk("setupConst: c_perm");
  }

  static void setup_1d()
  {
    ErrorCheck ec;
    cudaMemcpyToSymbol(c_perm_1, perm_1, sizeof(unsigned char) * 4, 0); 
    ec.chk("setupConst: c_perm_1");
  }
};


} //namespace 

#endif
