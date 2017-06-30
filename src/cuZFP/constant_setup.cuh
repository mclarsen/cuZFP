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

    const uint max_prec = get_precision<Scalar>();
    cudaMemcpyToSymbol(c_maxprec, &max_prec, sizeof(uint)); 
    ec.chk("setupConst: c_maxprec");
    
    const int min_exp  = get_min_exp<Scalar>();
    cudaMemcpyToSymbol(c_minexp, &min_exp, sizeof(int)); 
    ec.chk("setupConst: c_minexp");
    
    const int ebits = get_ebits<Scalar>();
    cudaMemcpyToSymbol(c_ebits, &ebits, sizeof(int)); 
    ec.chk("setupConst: c_ebits");

    const int ebias = get_ebias<Scalar>();
    cudaMemcpyToSymbol(c_ebias, &ebias, sizeof(int)); 
    ec.chk("setupConst: c_ebias");

  }
};

} //namespace 

#endif
