#include <assert.h>
#include "cuZFP.h"
#include "encode.cuh"
#include "ErrorCheck.h"
#include "decode.cuh"
#include <thrust/device_vector.h>
#include <iostream>

#define BSIZE  16
uint MAXBITS = BSIZE*64;
uint MAXPREC = 64;
int MINEXP = -1074;
//const double rate = BSIZE;
//size_t  blksize = 0;
//uint size = 64;
int EBITS = 11;                     /* number of exponent bits */
const int EBIAS = 1023;
//const int intprec = 64;
namespace cuZFP {

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))
static const unsigned char
perm[64] = {
	index(0, 0, 0), //  0 : 0

	index(1, 0, 0), //  1 : 1
	index(0, 1, 0), //  2 : 1
	index(0, 0, 1), //  3 : 1

	index(0, 1, 1), //  4 : 2
	index(1, 0, 1), //  5 : 2
	index(1, 1, 0), //  6 : 2

	index(2, 0, 0), //  7 : 2
	index(0, 2, 0), //  8 : 2
	index(0, 0, 2), //  9 : 2

	index(1, 1, 1), // 10 : 3

	index(2, 1, 0), // 11 : 3
	index(2, 0, 1), // 12 : 3
	index(0, 2, 1), // 13 : 3
	index(1, 2, 0), // 14 : 3
	index(1, 0, 2), // 15 : 3
	index(0, 1, 2), // 16 : 3

	index(3, 0, 0), // 17 : 3
	index(0, 3, 0), // 18 : 3
	index(0, 0, 3), // 19 : 3

	index(2, 1, 1), // 20 : 4
	index(1, 2, 1), // 21 : 4
	index(1, 1, 2), // 22 : 4

	index(0, 2, 2), // 23 : 4
	index(2, 0, 2), // 24 : 4
	index(2, 2, 0), // 25 : 4

	index(3, 1, 0), // 26 : 4
	index(3, 0, 1), // 27 : 4
	index(0, 3, 1), // 28 : 4
	index(1, 3, 0), // 29 : 4
	index(1, 0, 3), // 30 : 4
	index(0, 1, 3), // 31 : 4

	index(1, 2, 2), // 32 : 5
	index(2, 1, 2), // 33 : 5
	index(2, 2, 1), // 34 : 5

	index(3, 1, 1), // 35 : 5
	index(1, 3, 1), // 36 : 5
	index(1, 1, 3), // 37 : 5

	index(3, 2, 0), // 38 : 5
	index(3, 0, 2), // 39 : 5
	index(0, 3, 2), // 40 : 5
	index(2, 3, 0), // 41 : 5
	index(2, 0, 3), // 42 : 5
	index(0, 2, 3), // 43 : 5

	index(2, 2, 2), // 44 : 6

	index(3, 2, 1), // 45 : 6
	index(3, 1, 2), // 46 : 6
	index(1, 3, 2), // 47 : 6
	index(2, 3, 1), // 48 : 6
	index(2, 1, 3), // 49 : 6
	index(1, 2, 3), // 50 : 6

	index(0, 3, 3), // 51 : 6
	index(3, 0, 3), // 52 : 6
	index(3, 3, 0), // 53 : 6

	index(3, 2, 2), // 54 : 7
	index(2, 3, 2), // 55 : 7
	index(2, 2, 3), // 56 : 7

	index(1, 3, 3), // 57 : 7
	index(3, 1, 3), // 58 : 7
	index(3, 3, 1), // 59 : 7

	index(2, 3, 3), // 60 : 8
	index(3, 2, 3), // 61 : 8
	index(3, 3, 2), // 62 : 8

	index(3, 3, 3), // 63 : 9
};
template<class Scalar>
void setupConst(const unsigned char *perm,
                uint maxbits_,
                uint maxprec_,
                int minexp_,
                int ebits_,
                int ebias_)
{
	ErrorCheck ec;
	ec.chk("setupConst start");
	cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char) * 64, 0); ec.chk("setupConst: c_perm");

	cudaMemcpyToSymbol(c_maxbits, &MAXBITS, sizeof(uint)); ec.chk("setupConst: c_maxbits");
	const uint sizeof_scalar = sizeof(Scalar);
	cudaMemcpyToSymbol(c_sizeof_scalar, &sizeof_scalar, sizeof(uint)); ec.chk("setupConst: c_sizeof_scalar");

	cudaMemcpyToSymbol(c_maxprec, &maxprec_, sizeof(uint)); ec.chk("setupConst: c_maxprec");
	cudaMemcpyToSymbol(c_minexp, &minexp_, sizeof(int)); ec.chk("setupConst: c_minexp");
	cudaMemcpyToSymbol(c_ebits, &ebits_, sizeof(int)); ec.chk("setupConst: c_ebits");
	cudaMemcpyToSymbol(c_ebias, &ebias_, sizeof(int)); ec.chk("setupConst: c_ebias");

	ec.chk("setupConst finished");



}

void encode(int nx, int ny, int nz, std::vector<double> &in_data, EncodedData &encoded_data)
{
  ErrorCheck errors;
  assert(in_data.size() == nx * ny * nz);
  thrust::device_vector<double> d_in_data(in_data); 
  const size_t bits_per_val = 16;// THIS is bits per value 
                                // total allocated space is (num_values / num_blocks) * 64 bits per word /
  const size_t total_blocks = in_data.size() / 64; 
  const size_t bits_per_word = 64;
  const size_t total_bits = bits_per_val * in_data.size();
  const size_t alloc_size = total_bits / bits_per_word;
  thrust::device_vector<Word> d_encoded;
  d_encoded.resize(alloc_size);
  std::cout<<"Encoding\n";
	setupConst<double>(perm, MAXBITS, MAXPREC, MINEXP, EBITS, EBIAS);
  encode<long long int, unsigned long long, double, 16, 64>(nx, ny, nz, d_in_data, d_encoded, 64); 
  errors.chk("Encode");
  encoded_data.m_data.resize(d_encoded.size());

  //thrust::copy(encoded_data.m_data.begin(), 
  //             encoded_data.m_data.end(),
  //             d_encoded.begin());
  Word * d_ptr = thrust::raw_pointer_cast(d_encoded.data());
  Word * h_ptr = &encoded_data.m_data[0];
  cudaMemcpy(h_ptr, d_ptr, alloc_size * sizeof(Word), cudaMemcpyDeviceToHost);
  encoded_data.m_dims[0] = nx;
  encoded_data.m_dims[1] = ny;
  encoded_data.m_dims[2] = nz;
}

void decode(const EncodedData &encoded_data, std::vector<double> &out_data)
{
  const int nx = encoded_data.m_dims[0];
  const int ny = encoded_data.m_dims[1];
  const int nz = encoded_data.m_dims[2];
  const size_t out_size = nx * ny * nz;

  thrust::device_vector<double> d_out_data(out_size); 
  thrust::device_vector<Word> d_encoded(encoded_data.m_data);

  decode<long long int, unsigned long long, double, 16, 64>(nx, ny, nz, d_encoded, d_out_data); 

  out_data.resize(out_size); 
  thrust::copy(d_out_data.begin(), 
               d_out_data.end(),
               out_data.begin());
}


} // namespace cuZFP

