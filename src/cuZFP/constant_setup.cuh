#ifndef cuZFP_CONSTANT_SETUP
#define cuZFP_CONSTANT_SETUP
#
#include <constants.h>
#include <shared.h>
#include <ErrorCheck.h>

namespace cuZFP {


inline __host__ __device__ int get_ebias(double) { return 1023; }
inline __host__ __device__ int get_ebias(float) { return 127; }

inline __host__ __device__ int get_ebits(double) { return 11; }
inline __host__ __device__ int get_ebits(float) { return 8; }

inline __host__ __device__ uint get_max_prec(double) { return 64; }
inline __host__ __device__ uint get_max_prec(float) { return 32; }

inline __host__ __device__ int get_min_exp(double) { return -1074; }

class ConstantSetup
{
public:
  template<typename Scalar>
  static void setup_3d(Scalar, const int rate)
  { 
    ErrorCheck ec;
    cudaMemcpyToSymbol(c_perm, perm_3d, sizeof(unsigned char) * 64, 0); 
    ec.chk("setupConst: c_perm");

    const int vals_per_block = 64;
    const uint max_bits = rate * vals_per_block; 
    cudaMemcpyToSymbol(c_maxbits, &max_bits, sizeof(uint)); 
    ec.chk("setupConst: c_maxbits");

    const uint sizeof_scalar = sizeof(Scalar);

    cudaMemcpyToSymbol(c_sizeof_scalar, &sizeof_scalar, sizeof(uint)); 
    ec.chk("setupConst: c_sizeof_scalar");

    const uint max_prec = get_max_prec(Scalar());
    cudaMemcpyToSymbol(c_maxprec, &max_prec, sizeof(uint)); 
    ec.chk("setupConst: c_maxprec");
    
    const int min_exp  = get_min_exp(Scalar());
    cudaMemcpyToSymbol(c_minexp, &min_exp, sizeof(int)); 
    ec.chk("setupConst: c_minexp");
    
    const int ebits = get_ebits(Scalar());
    cudaMemcpyToSymbol(c_ebits, &ebits, sizeof(int)); 
    ec.chk("setupConst: c_ebits");

    const int ebias = get_ebias(Scalar());
    cudaMemcpyToSymbol(c_ebias, &ebias, sizeof(int)); 
    ec.chk("setupConst: c_ebias");

  }
};

} //namespace 

#endif
