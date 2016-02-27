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
#include "include/decode.cuh"

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


template<class Scalar>
void setupConst(const unsigned char *perm)
{
	ErrorCheck ec;
	ec.chk("setupConst start");
	cudaMemcpyToSymbol(c_perm, perm, sizeof(unsigned char) * 64, 0); ec.chk("setupConst: c_perm");

	const uint sizeof_scalar = sizeof(Scalar);
	cudaMemcpyToSymbol(c_sizeof_scalar, &sizeof_scalar, sizeof(uint)); ec.chk("setupConst: c_sizeof_scalar");

	ec.chk("setupConst finished");


}


/* reorder unsigned coefficients and convert to signed integer */
template<class Int, class UInt>
__host__
static void
inv_order(const UInt* ublock, Int* iblock, const unsigned char* perm, uint n)
{
	do
		iblock[*perm++] = uint2int<UInt>(*ublock++);
	while (--n);
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

	for (int i = 0; i < stream.size(); i++){
		for (int j = 0; j < 64; j++){
			assert(stream[i].begin[j] == *ptr++);

		}
	}

}

template<class Scalar>
void validateEncode
(
host_vector<Scalar> &p,
BitStream *stream
)
{

}

template<class UInt, uint bsize>
__device__ __host__
uint
decode_ints_par(Bit<bsize> & stream, UInt* data, uint minbits, uint maxbits, uint maxprec, unsigned long long orig_count, uint size)
{


  unsigned long long count = orig_count;
    const uint intprec = CHAR_BIT * (uint)sizeof(UInt);
    const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
    uint bits = maxbits;

    /* initialize data array to all zeros */
    for (uint i = 0; i < size; i++)
      data[i] = 0;

    uint tmp_g[64], tmp_n[64], cnt_sft[64], bit_bits[64];
    char bit_offset[64];
    Word bit_buffer[64];

    for (int i=0; i<64; i++){
      bit_offset[i] = 0;
      bit_buffer[i] = 0;
      tmp_g[i] = tmp_n[i] = cnt_sft[i] = 0;
    }

    /* input one bit plane at a time from MSB to LSB */
    for (uint k = intprec, n = 0; k-- > kmin;) {
      /* decode bit plane k */
      tmp_n[k] = n;
      bit_bits[k] = stream.bits;
      bit_offset[k] = stream.offset;
      bit_buffer[k] = stream.buffer;
      for (uint m = n;;tmp_g[k]++) {

        if (bits){
          /* decode bit k for the next set of m values */
          m = MIN(m, bits);
          bits -= m;
          unsigned long long x = stream.read_bits(m);
          /* continue with next bit plane if there are no more groups */
          if (!count || !bits){

            break;
          }
          /* perform group test */
          bits--;
          uint test = stream.read_bit();
          /* continue with next bit plane if there are no more significant bits */
          if (!test || !bits){
            break;
          }
          /* decode next group of m values */
          m = count & 0xfu;
          count >>= 4;
          n += m;
        }
      }
    }

    /* read at least minbits bits */
    while (bits > maxbits - minbits) {
      bits--;
      stream.read_bit();
    }

    for (uint i = 0; i < size; i++)
      data[i] = 0;

    stream.rewind();
    bool first = true;

#pragma omp parallel for
    for (int q = 0; q<64; q++){
      Bit<bsize> cache[64];
      for (int i=0; i<64; i++){
        cache[i] = stream;
      }
      unsigned long long count = orig_count;
      uint new_bits = maxbits;
      for (uint k=intprec; k-- > kmin;){
        cache[k].seek(bit_offset[k], bit_bits[k], bit_buffer[k]);
        for (uint i=0, m = tmp_n[k], n = 0; i<tmp_g[k]+1; i++){
          if (new_bits){
            /* decode bit k for the next set of m values */
            m = MIN(m, new_bits);
            n += m;
            new_bits -= m;

//            if (first)
//              for (x = cache[k].read_bits(m); m; m--, x >>= 1)
//                data[n - m] += (UInt)(x & 1u) << k;

            unsigned long long x = cache[k].read_bits(m);
            x >>= q - (n-m);
            data[q] += (UInt)(x &1u) << k;

            /* continue with next bit plane if there are no more groups */
            if (!count || !new_bits)
              break;
            /* perform group test */
            new_bits--;
            uint test = cache[k].read_bit();
            /* cache[k] with next bit plane if there are no more significant bits */
            if (!test || !new_bits)
              break;
            /* decode next group of m values */
            m = count & 0xfu;
            count >>= 4;
          }
        }
      }
      first = false;
    }
//    /* read at least minbits bits */
//    while (new_bits > maxbits - minbits) {
//      new_bits--;
//      stream.read_bit();
//    }

    return maxbits - bits;
}

template<class Scalar>
void
gather(Scalar* q, const Scalar* p, uint mx, uint my, uint mz, uint sx, uint sy, uint sz)
{
	for (int z = mz; z < mz + 4; z++)
		for (int y = my; y < my + 4; y++)
			for (int x = mx; x < mx + 4; x++, q++)
				*q = p[z*sz + y*sy + x*sx];
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
	int chk_idx = 0;
	for (int z = 0; z < nz; z += 4){
		for (int y = 0; y < ny; y += 4){
			for (int x = 0; x < nx; x += 4){
				int idx = z*nx*ny + y*nx + x;
				Int q1[64], q2[64], q3[64];
				UInt buf[64];

				int emax2 = max_exp<Scalar>(raw_pointer_cast(p.data()), x, y, z, 1, nx, nx*ny);
				fixed_point(q1, raw_pointer_cast(p.data()), emax2, x, y, z, 1, nx, nx*ny);
				for (int i = 0; i < 64; i++){
					q2[i] = q1[i];
				}

				fwd_xform<Int>(q2);
				for (int i = 0; i < 64; i++){
					q3[i] = q2[i];
				}
				reorder<Int, UInt>(q3, buf);

				encode_ints_old<UInt>(stream_old, buf, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

				stream_old->rewind();
				UInt dec[64];
				decode_ints_old<Int, UInt>(stream_old, minbits, maxbits, precision(emax2, maxprec, minexp), dec, size, group_count);

				for (int i = 0; i < 64; i++){
					//assert(dec[i] == buf[i]);
					if (dec[i] != buf[i]){
						cout << "old decode failed: " << i << " " << dec[i] << " " << buf[i] << endl;
						exit(-1);

					}

				}

				Int iblock[64];
				inv_order(dec, iblock, perm, 64);
				for (int i = 0; i < 64; i++){
					assert(iblock[i] == q2[i]);
				}
				inv_xform(iblock);
				for (int i = 0; i < 64; i++){
					assert(iblock[i] == q1[i]);
				}
				Scalar fblock[64], cblock[64];
				gather<Scalar>(cblock, raw_pointer_cast(p.data()), x, y, z, 1, nx, nx*ny);

				inv_cast<Int, Scalar, sizeof(Scalar)>(iblock, fblock, emax2, 0, 0, 0, 1, 4, 16);
				for (int i = 0; i < 64; i++){
					assert(FABS(fblock[i] - cblock[i]) < 1e-6);
				}
				chk_idx++;
			}
		}
	}


	double start_time = omp_get_wtime();
#pragma omp parallel for
	for (int z = 0; z < nz; z += 4){
		for (int y = 0; y < ny; y += 4){
			for (int x = 0; x < nx; x += 4){
				int idx = z*nx*ny + y*nx + x;
				Int q2[64];
				UInt buf[64];

				int emax2 = max_exp<Scalar>(raw_pointer_cast(p.data()), x, y, z, 1, nx, nx*ny);
				fixed_point(q2, raw_pointer_cast(p.data()), emax2, x, y, z, 1, nx, nx*ny);
				fwd_xform<Int>(q2);
				reorder<Int, UInt>(q2, buf);
				encode_ints<UInt>(stream[z / 4 * mx*my + y / 4 * mx + x / 4], buf, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

				stream[z / 4 * mx*my + y / 4 * mx + x / 4].rewind();
				UInt dec[64];
				decode_ints<UInt, bsize>(stream[z / 4 * mx*my + y / 4 * mx + x / 4], dec, minbits, maxbits, maxprec, group_count, size);
				Int iblock[64];
				inv_order(dec, iblock, perm, 64);
				//                for (int i=0; i<64; i++){
				//                    assert(iblock[i] == q2[i]);
				//                }
				inv_xform(iblock);
				//                for (int i=0; i<64; i++){
				//                    assert(iblock[i] == q1[i]);
				//                }
				Scalar fblock[64], cblock[64];
				gather<Scalar>(cblock, raw_pointer_cast(p.data()), x, y, z, 1, nx, nx*ny);
				inv_cast<Int, Scalar, sizeof(Scalar)>(iblock, fblock, emax2, 0, 0, 0, 1, 4, 16);
				for (int i = 0; i < 64; i++){
					assert(FABS(fblock[i] - cblock[i]) < 1e-6);
				}

			}
		}
	}

	double elapsed_time = omp_get_wtime() - start_time;
	cout << "CPU elapsed time: " << elapsed_time << endl;
	//validateCPU(stream_old, stream);
	//validateEncode(stream_old, p);
}

template<class Int, class UInt, class Scalar, uint bsize>
void gpuValidate
(
host_vector<Scalar> &h_p,
device_vector<Int> &q,
device_vector<Scalar> &data
)
{
	host_vector<Int> h_q;

	h_q = q;

  uint x,y,z;
  x=y=z=0;

  host_vector<Int> q2(64);
  host_vector<UInt> buf(64);
  Bit<bsize> loc_stream;
  int emax2 = max_exp<Scalar>(raw_pointer_cast(h_p.data()), x, y, z, 1, nx, nx*ny);
  fixed_point(raw_pointer_cast(q2.data()), raw_pointer_cast(h_p.data()), emax2, x, y, z, 1, nx, nx*ny);
  fwd_xform(raw_pointer_cast(q2.data()));
  reorder<Int, UInt>(raw_pointer_cast(q2.data()), raw_pointer_cast(buf.data()));
  encode_ints<UInt>(loc_stream, raw_pointer_cast(buf.data()), minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

  loc_stream.rewind();
  UInt dec1[64], dec2[64];

  decode_ints<UInt, bsize>(loc_stream, dec1, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

  loc_stream.rewind();
  decode_ints_par<UInt, bsize>(loc_stream, dec2, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);
  for (int j = 0; j < 64; j++){
    if(dec1[j] != dec2[j]){
      cout << "parallel failed: " << j << " " << dec1[j] << " " << dec2[j] << endl;
      //exit(-1);
    }
  }


//	int i = 0;
//	for (int z = 0; z < nz; z += 4){
//		for (int y = 0; y < ny; y += 4){
//			for (int x = 0; x < nx; x += 4){
//				int idx = z*nx*ny + y*nx + x;
//				host_vector<Int> q2(64);
//				host_vector<UInt> buf(64);
//				Bit<bsize> loc_stream;
//				int emax2 = max_exp<Scalar>(raw_pointer_cast(h_p.data()), x, y, z, 1, nx, nx*ny);
//				fixed_point(raw_pointer_cast(q2.data()), raw_pointer_cast(h_p.data()), emax2, x, y, z, 1, nx, nx*ny);
//				fwd_xform(raw_pointer_cast(q2.data()));
//				reorder<Int, UInt>(raw_pointer_cast(q2.data()), raw_pointer_cast(buf.data()));
//				encode_ints<UInt>(loc_stream, raw_pointer_cast(buf.data()), minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

//				loc_stream.rewind();
//        UInt dec1[64], dec2[64];

//        decode_ints<UInt, bsize>(loc_stream, dec1, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);

//        loc_stream.rewind();
//        decode_ints_par<UInt, bsize>(loc_stream, dec2, minbits, maxbits, precision(emax2, maxprec, minexp), group_count, size);
//				for (int j = 0; j < 64; j++){
//          if(dec1[j] != dec2[j]){
//            cout << "parallel failed: " << z << " " << y << " " << x << " " << j << " " << dec1[j] << " " << dec2[j] << endl;
//            exit(-1);
//          }
//				}

////				Int iblock[64];
////				inv_order(dec, iblock, perm, 64);
////				inv_xform(iblock);

////				for (int j = 0; j < 64; j++){
////					assert(h_q[i * 64 + j] == iblock[j]);
////				}

////				Scalar fblock[64];
////				inv_cast(iblock, fblock, emax2, 0, 0, 0, 1, 4, 16);

////				int fidx = 0;
////				for (int k = z; k < z+4; k++){
////					for (int j = y; j < y + 4; j++){
////						for (int i = x; i < x + 4; i++, fidx++){
////              if (h_p[k*nz*ny + j*ny + i] != fblock[fidx]){
////                cout << "inv_cast failed: " << k << " " << j << " " << i << " " << fidx << " " << h_p[k*nz*ny + j*ny + i] << " " << fblock[fidx] << endl;
////								exit(-1);
////							}

////						}
////					}
////				}
//				i++;

//			}
//		}
//	}
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

	host_vector<Scalar> h_data;
	h_data = data;

	dim3 emax_size(nx / 4, ny / 4, nz / 4);

	dim3 block_size(8, 8, 8);
	dim3 grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	device_vector<int> emax(emax_size.x * emax_size.y * emax_size.z);

	ErrorCheck ec;

	cudaEvent_t start, stop;
	float millisecs;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	ec.chk("pre-cudaMaxExp");
	cudaMaxExp << <grid_size, block_size >> >
		(
		raw_pointer_cast(emax.data()),
		raw_pointer_cast(data.data())
		);
	ec.chk("cudaMaxExp");

	ec.chk("pre-cudaFixedPoint");
	cudaFixedPoint << <grid_size, block_size >> >
		(
		raw_pointer_cast(emax.data()),
		raw_pointer_cast(data.data()),
		raw_pointer_cast(q.data())
		);
	ec.chk("cudaFixedPoint");


	cudaDecorrelate<Int> << <grid_size, block_size >> >
		(
		raw_pointer_cast(q.data())
		);
	ec.chk("cudaDecorrelate");

	block_size = dim3(8, 8, 8);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaint2uint << < grid_size, block_size >> >
		(
		raw_pointer_cast(q.data()),
		raw_pointer_cast(buffer.data())
		);
	ec.chk("cudaint2uint");

	//    q.clear();
	//    q.shrink_to_fit();

	device_vector<Bit<bsize> > stream(emax_size.x * emax_size.y * emax_size.z);


	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaencode<UInt, bsize> << < emax_size.x*emax_size.y*emax_size.z / 16, 16, 16 * (sizeof(Bit<bsize>) + sizeof(int)) >> >
		(
		raw_pointer_cast(buffer.data()),
		raw_pointer_cast(stream.data()),
		raw_pointer_cast(emax.data()),
		minbits, maxbits, maxprec, minexp, group_count, size
		);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs << endl;


	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaRewind<bsize> << < grid_size, block_size >> >
		(
		raw_pointer_cast(stream.data())
		);
	ec.chk("cudaRewind");

	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaDecode<UInt, bsize> << < emax_size.x*emax_size.y*emax_size.z / 16, 16, 16 * (sizeof(Bit<bsize>) + sizeof(int)) >> >
		(
		raw_pointer_cast(buffer.data()),
		raw_pointer_cast(stream.data()),
		raw_pointer_cast(emax.data()),
		minbits, maxbits, maxprec, minexp, group_count, size
		);
	cudaStreamSynchronize(0);
	ec.chk("cudaDecode");

	block_size = dim3(8, 8, 8);
	grid_size = dim3(nx, ny, nz);
	grid_size.x /= block_size.x; grid_size.y /= block_size.y; grid_size.z /= block_size.z;
	cudaInvOrder << <grid_size, block_size >> >
		(
		raw_pointer_cast(buffer.data()),
		raw_pointer_cast(q.data())
		);
	cudaStreamSynchronize(0);
	ec.chk("cudaInvOrder");

	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	cudaInvXForm<Int> << <grid_size, block_size >> >
		(
		raw_pointer_cast(q.data())
		);
	cudaStreamSynchronize(0);
	ec.chk("cudaInvXForm");

	block_size = dim3(8, 8, 8);
	grid_size = emax_size;
	grid_size.x /= block_size.x; grid_size.y /= block_size.y;  grid_size.z /= block_size.z;

	cudaInvCast<Int, Scalar> << <grid_size, block_size >> >
		(
		raw_pointer_cast(emax.data()),
		raw_pointer_cast(data.data()),
		raw_pointer_cast(q.data())
		);
	ec.chk("cudaInvCast");

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millisecs, start, stop);
	ec.chk("cudaencode");

	cout << "encode GPU in time: " << millisecs << endl;

	gpuValidate<Int, UInt, Scalar, bsize>(h_data, q, data);

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
	setupConst<double>(perm);
//	cout << "Begin gpuTestBitStream" << endl;
//	gpuTestBitStream<long long, unsigned long long, double, 64>(d_vec_in, d_vec_out, d_vec_buffer);
//	cout << "Finish gpuTestBitStream" << endl;
	//    cout << "Begin cpuTestBitStream" << endl;
	//    cpuTestBitStream<long long, unsigned long long, double, 64>(h_vec_in);
	//    cout << "End cpuTestBitStream" << endl;

  gpuValidate<long long, unsigned long long, double, 64>(h_vec_in, d_vec_out, d_vec_in);
}
