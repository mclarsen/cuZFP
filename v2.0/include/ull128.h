__device__ __host__
ulonglong2 lshiftull2(const ulonglong2 &in, size_t len)
{
	ulonglong2 a = in;
	unsigned long long value = a.x;
	if (len < 64){
		unsigned long long v = value >> (64-len);
		a.y <<= len;
		a.y += v;

		a.x <<= len;
	}
	else{
		len -= 64;
		unsigned long long v = value >> len;

		a.y <<= len;
		a.y += v;
	}

	return a;
}
__device__ __host__
ulonglong2 rshiftull2(const ulonglong2 &in, size_t len)
{
	ulonglong2 a = in;
	unsigned long long value = a.y;
	if (len < 64){
		a.x >>= len;
		value <<= (64 - len);
		a.x += value;

		a.y >>= len;
	}
	else{
		a.y >>= (len - 64);
		a.x = a.y;
		a.y = 0;
	}

	return a;
}