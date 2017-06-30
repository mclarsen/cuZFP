#ifndef cuZFP_TYPE_INFO
#define cuZFP_TYPE_INFO

namespace cuZFP {

template<typename T> inline __host__ __device__ int get_ebias();
template<> inline __host__ __device__ int get_ebias<double>() { return 1023; }
template<> inline __host__ __device__ int get_ebias<float>() { return 127; }

template<typename T> inline __host__ __device__ int get_ebits();
template<> inline __host__ __device__ int get_ebits<double>() { return 11; }
template<> inline __host__ __device__ int get_ebits<float>() { return 8; }

template<typename T> inline __host__ __device__ int get_precision();
template<> inline __host__ __device__ int get_precision<double>() { return 64; }
template<> inline __host__ __device__ int get_precision<long long int>() { return 64; }
template<> inline __host__ __device__ int get_precision<float>() { return 32; }
template<> inline __host__ __device__ int get_precision<int>() { return 32; }

template<typename T> inline __host__ __device__ int get_min_exp();
template<> inline __host__ __device__ int get_min_exp<double>() { return -1074; }
template<> inline __host__ __device__ int get_min_exp<float>() { return -1074; }

template<typename T> inline __host__ __device__ int scalar_sizeof();

template<> inline __host__ __device__ int scalar_sizeof<double>() { return 8; }
template<> inline __host__ __device__ int scalar_sizeof<long long int>() { return 8; }
template<> inline __host__ __device__ int scalar_sizeof<float>() { return 4; }
template<> inline __host__ __device__ int scalar_sizeof<int>() { return 4; }

template<typename T> struct zfp_traits;

template<> struct zfp_traits<double>
{
  typedef unsigned long long int UInt;
  typedef long long int Int;
};

template<> struct zfp_traits<float>
{
  typedef unsigned int UInt;
  typedef int Int;
};

} // namespace cuZFP
#endif
