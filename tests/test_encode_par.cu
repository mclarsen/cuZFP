#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <omp.h>

#define KEPLER 0
#include "ErrorCheck.h"
#include "include/encode.cuh"
#include "include/BitStream.cuh"

using namespace thrust;
using namespace std;

#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

const size_t nx = 128;
const size_t ny = 128;
const size_t nz = 128;

uint minbits = 4096;
uint maxbits = 4096;
uint maxprec = 64;
int minexp = -1074;
const double rate = 64;
size_t  blksize = 0;
unsigned long long group_count = 0x46acca631ull;
uint size = 64;


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


static size_t block_size(double rate) { return (lrint(64 * rate) + CHAR_BIT - 1) / CHAR_BIT; }


void setupConst(const unsigned char *perm)
{
    ErrorCheck ec;
    ec.chk("setupConst start");
    cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char)*64,0); ec.chk("setupConst: lic_dim");
    ec.chk("setupConst finished");


}




//Used to generate rand array in CUDA with Thrust
struct RandGen
{
    RandGen() {}

    __device__ float operator () (const uint idx)
    {
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist(0.0, 0.0001);
        randEng.discard(idx);
        return uniDist(randEng);
    }
};
template<class Int, class UInt>
void reorder
(
        const Int *q,
        UInt *buffer
        )
{
    for (uint i = 0; i < 64; i++)
      buffer[i] = int2uint<Int, UInt>(q[perm[i]]);
}


template<uint bsize>
void validateCPU
(
        const BitStream *stream_old,
        host_vector<Bit<bsize> > &stream
        )
{
    Word *ptr = stream_old->begin;

    for (int i=0; i < stream.size(); i++){
        for (int j=0; j<64; j++){
            assert(stream[i].begin[j] == *ptr++);

        }
    }

}
void
__device__ __host__
write_bitters(Word &bitters, unsigned long long value, uint n, uint &sbits)
{
  unsigned long long v = value >> n;
  value -= v << n;
  bitters += value << sbits;
  sbits += n;
}

__device__ __host__
void
write_bitter(Word &bitters, uint bit, uint &sbits)
{
  bitters += (Word)bit << sbits++;
}
template<class UInt, uint bsize>
__device__ __host__
void encode_bit_plane_par(const unsigned long long *x, const uint *g, Bit<bsize> & stream, uint minbits, uint maxbits, uint maxprec, unsigned long long count)
{
    uint m, n;
   uint k = 0;
   uint kmin = intprec > maxprec ? intprec - maxprec : 0;
   uint bits = maxbits;

   /* serial: output one bit plane at a time from MSB to LSB */
   for (k = intprec, n = 0; k-- > kmin;) {
    /* encode bit k for first n values */
    unsigned long long y = x[k];
    if (n < bits) {
      y = stream.write_bits(y, n);
      bits -= n;
    }
    else {
      stream.write_bits(y, bits);
      bits = 0;
      return;
    }
    uint h = g[min(k+1,intprec-1)];
    /* perform series of group tests */
    while (h++ < g[k]) {
      /* output a one bit for a positive group test */
      stream.write_bit(1);
      bits--;
      /* add next group of m values to significant set */
      m = count & 0xfu;
      count >>= 4;
      n += m;
      /* encode next group of m values */
      if (m < bits) {
        y = stream.write_bits( y, m);
        bits -= m;
      }
      else {
        stream.write_bits(y, bits);
        bits = 0;
        return;
      }
    }
    /* if there are more groups, output a zero bit for a negative group test */
    if (count) {
      stream.write_bit(0);
      bits--;
    }
   }

   /* write at least minbits bits by padding with zeros */
   while (bits > maxbits - minbits) {
    stream.write_bit(0);
    bits--;
   }
}
template<class UInt, uint bsize>
__device__ __host__
void encode_bit_plane_thrust(const unsigned long long *x, const uint *g, Word *bitters, Word *out, uint *sbits, Bit<bsize> & stream, uint minbits, uint maxbits, uint maxprec, unsigned long long count)
{
	uint m, n;
	uint k = 0;
	uint kmin = intprec > maxprec ? intprec - maxprec : 0;
	uint bits = maxbits;

	/* serial: output one bit plane at a time from MSB to LSB */
	for (k = intprec, n = 0; k-- > kmin;) {
		bitters[(intprec - 1) - k] = 0;
		sbits[(intprec - 1) - k] = 0;
		/* encode bit k for first n values */
		unsigned long long y = x[k];
		if (n < bits) {
			bits -= n;
			write_bitters(bitters[(intprec - 1) - k], y, n, sbits[(intprec - 1) - k]);
		}
		else {
			bits = 0;
			return;
		}

		uint h = g[min(k + 1, intprec - 1)];
		/* perform series of group tests */
		while (h++ < g[k]) {
			/* output a one bit for a positive group test */
			write_bitter(bitters[(intprec - 1) - k], 1, sbits[(intprec - 1) - k]);
			bits--;
			/* add next group of m values to significant set */
			m = count & 0xfu;
			count >>= 4;
			n += m;
			/* encode next group of m values */
			if (m < bits) {
				write_bitters(bitters[(intprec - 1) - k], y, m, sbits[(intprec - 1) - k]);
				bits -= m;
			}
			else {
				write_bitters(bitters[(intprec - 1) - k], y, m, sbits[(intprec - 1) - k]);
				bits = 0;
				return;
			}
		}
		/* if there are more groups, output a zero bit for a negative group test */
		if (count) {
			write_bitter(bitters[(intprec - 1) - k], 0, sbits[(intprec - 1) - k]);
			bits--;
		}
	}

	uint tot_sbits = sbits[0];
	uint cur_out = 0;
	out[cur_out] = 0;

	for (int i = 0; i < CHAR_BIT *sizeof(UInt); i++){
		unsigned long long value = bitters[i];
		unsigned long long v = value >> sbits[i];
		value -= v << sbits[i];
		out[cur_out] += value << tot_sbits;
		tot_sbits += sbits[i];
		if (tot_sbits >= wsize){
			tot_sbits -= wsize;
			cur_out++;
			out[cur_out] = bitters[i] >> (sbits[i] - tot_sbits);
		}
	}

#ifndef __CUDA_ARCH__
   cout << "tot bits: " << tot_sbits << " bits: " << bits << endl;
	 for (int i = 0; i < CHAR_BIT*sizeof(UInt); i++){
		 if (out[i] != stream.begin[i]){
			 cout << "failed: " << i << " " << out[i] << " " << stream.begin[i] << endl;
			 exit(-1);
		 }
	 }

#endif
}
template<class UInt, uint bsize>
static void
encode_ints_par(Bit<bsize> & stream, const UInt* data, uint prec)
{
    uint kmin = intprec > maxprec ? intprec - maxprec : 0;

    unsigned long long x[CHAR_BIT * sizeof(UInt)];
    uint g[CHAR_BIT * sizeof(UInt)];


    encode_group_test<UInt, bsize>(x, g, data, minbits, maxbits, prec, group_count, size);
    uint cur = g[CHAR_BIT * sizeof(UInt)-1];
    uint k = intprec-1;
    for (k = intprec-1; k-- > kmin;) {
        if (cur < g[k])
            cur = g[k];
        else if (cur > g[k])
            g[k] = cur;
    }
    encode_bit_plane_par<UInt, bsize>(x, g, stream, minbits, maxbits, prec, group_count);

		Word bitters[CHAR_BIT * sizeof(UInt)], out[CHAR_BIT * sizeof(UInt)];
	uint sbits[CHAR_BIT *sizeof(UInt)];

	for (int i = 0; i < CHAR_BIT *sizeof(UInt); i++){
		out[i] = 0;
	}

	encode_bit_plane_thrust<UInt, bsize>(x, g, bitters, out, sbits, stream, minbits, maxbits, prec, group_count);

}

template<class Int, class UInt, class Scalar, uint bsize>
void cpuTestBitStream
(
        host_vector<Scalar> &p
        )
{
    blksize = block_size(rate);
    uint mx = nx / 4;
    uint my = ny / 4;
    uint mz = nz / 4;

    //BitStream *stream_old = stream_create(blksize*mx*my);
    BitStream *stream_old = stream_create(nx*ny*nz);
    host_vector<Bit<bsize> > stream(mx*my*mz);

    for (int z=0; z<nz; z+=4){
        for (int y=0; y<ny; y+=4){
            for (int x=0; x<nx; x+=4){
                int idx = z*nx*ny + y*nx + x;
                Int q2[64];
                UInt buf[64];

                int emax2 = max_exp<Scalar>(raw_pointer_cast(p.data()), x,y,z, 1,nx,nx*ny);
                fixed_point(q2,raw_pointer_cast(p.data()), emax2, x,y,z, 1,nx,nx*ny);
                fwd_xform<Int>(q2);
                reorder<Int, UInt>(q2, buf);
                encode_ints_old<UInt>(stream_old, buf, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);
                //encode_ints_old_par<UInt>(stream_old, buf, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

            }
        }
    }
    double start_time = omp_get_wtime();
    for (int z=0; z<nz; z+=4){
        for (int y=0; y<ny; y+=4){
            for (int x=0; x<nx; x+=4){
                int idx = z*nx*ny + y*nx + x;
                Int q2[64];
                UInt buf[64];

                int emax2 = max_exp<Scalar>(raw_pointer_cast(p.data()), x,y,z, 1,nx,nx*ny);
                fixed_point(q2,raw_pointer_cast(p.data()), emax2, x,y,z, 1,nx,nx*ny);
                fwd_xform<Int>(q2);
                reorder<Int, UInt>(q2, buf);
                encode_ints_par<UInt>(stream[z/4 * mx*my + y/4 *mx + x/4], buf, precision(emax2, maxprec, minexp));
            }
        }
    }

    double elapsed_time = omp_get_wtime() - start_time;
    cout << "CPU elapsed time: " <<  elapsed_time << endl;
    validateCPU(stream_old, stream);
}

template<class Int, class UInt, class Scalar, uint bsize>
void gpuValidate
(
        device_vector<Scalar> &data,
        device_vector<UInt> &buffer,
        device_vector<Bit<bsize> > &stream
        )
{
    host_vector<Scalar> h_p;
    host_vector<UInt> h_buf;
    host_vector<Bit<bsize> > h_bits;

    h_p = data;
    h_buf = buffer;
    h_bits = stream;

    int i=0;
    for (int z=0; z<nz; z+=4){
        for (int y=0; y<ny; y+=4){
            for (int x=0; x<nx; x+=4){
                int idx = z*nx*ny + y*nx + x;
                host_vector<Int> q2(64);
                host_vector<UInt> buf(64);
                Bit<bsize> loc_stream;
                int emax2 = max_exp<Scalar>(raw_pointer_cast(h_p.data()), idx, 1,nx,nx*ny);
                fixed_point(raw_pointer_cast(q2.data()),raw_pointer_cast(h_p.data()), emax2, idx, 1,nx,nx*ny);
                fwd_xform(raw_pointer_cast(q2.data()));
                reorder<Int, UInt>(raw_pointer_cast(q2.data()), raw_pointer_cast(buf.data()));
                encode_ints<UInt>(loc_stream,  raw_pointer_cast(buf.data()), minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

                for (int j=0; j<64; j++){
                    assert(h_bits[i].begin[j] == loc_stream.begin[j]);
                }

                i++;

            }
        }
    }
}


template<class Int, class UInt, class Scalar, uint bsize>
void gpuTestBitStream
(
        device_vector<Scalar> &data,
        device_vector<Int> &q,
        device_vector<UInt> &buffer
        )
{
    host_vector<int> h_emax;
    host_vector<Scalar> h_p;
    host_vector<Int> h_q;
    host_vector<UInt> h_buf;
    host_vector<Bit<bsize> > h_bits;

    dim3 emax_size(nx/4, ny/4, nz/4 );

    dim3 block_size(8,8,8);
    dim3 grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

    device_vector<int> emax(emax_size.x * emax_size.y * emax_size.z);

    ErrorCheck ec;

    cudaEvent_t start, stop;
    float millisecs;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord( start, 0 );


    ec.chk("pre-cudaMaxExp");
    cudaMaxExp<<<grid_size, block_size>>>
            (
                raw_pointer_cast(emax.data()),
                raw_pointer_cast(data.data())
                );
    ec.chk("cudaMaxExp");

    ec.chk("pre-cudaFixedPoint");
    cudaFixedPoint<<<grid_size, block_size>>>
            (
                raw_pointer_cast(emax.data()),
                raw_pointer_cast(data.data()),
                raw_pointer_cast(q.data())
                );
    ec.chk("cudaFixedPoint");


    cudaDecorrelate<Int><<<grid_size, block_size>>>
        (
            raw_pointer_cast(q.data())
            );
    ec.chk("cudaDecorrelate");

    block_size = dim3(8,8,8);
    grid_size = dim3(nx,ny,nz);
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    cudaint2uint<<< grid_size, block_size>>>
            (
                raw_pointer_cast(q.data()),
                raw_pointer_cast(buffer.data())
                );
    ec.chk("cudaint2uint");

    q.clear();
    q.shrink_to_fit();

    device_vector<Bit<bsize> > stream(emax_size.x * emax_size.y * emax_size.z);


    block_size = dim3(8,8,8);
    grid_size = emax_size;
    grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
    cudaencode<UInt, bsize><<< emax_size.x*emax_size.y*emax_size.z/16, 16, 16*(sizeof(Bit<bsize>) + sizeof(int))>>>
            (
                raw_pointer_cast(buffer.data()),
                raw_pointer_cast(stream.data()),
                raw_pointer_cast(emax.data()),
                minbits, maxbits, maxprec, minexp, group_count, size
                );

    cudaEventRecord( stop, 0 );
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &millisecs, start, stop );
    ec.chk("cudaencode");

    cout << "encode GPU in time: " << millisecs << endl;

    gpuValidate<Int, UInt, Scalar, bsize>(data, buffer, stream);

}

int main()
{

    device_vector<double> d_vec_in(nx*ny*nz);
    device_vector<long long> d_vec_out(nx*ny*nz);
    device_vector<unsigned long long> d_vec_buffer(nx*ny*nz);
    host_vector<double> h_vec_in;

    thrust::counting_iterator<uint> index_sequence_begin(0);
    thrust::transform(
                    index_sequence_begin,
                    index_sequence_begin + nx*ny*nz,
                    d_vec_in.begin(),
                    RandGen());

    h_vec_in = d_vec_in;
//    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
    setupConst(perm);
    cout << "Begin gpuTestBitStream" << endl;
    //gpuTestBitStream<long long, unsigned long long, double, 64>(d_vec_in, d_vec_out, d_vec_buffer);
    cout << "Finish gpuTestBitStream" << endl;
    cout << "Begin cpuTestBitStream" << endl;
    cpuTestBitStream<long long, unsigned long long, double, 64>(h_vec_in);
    cout << "End cpuTestBitStream" << endl;
}
