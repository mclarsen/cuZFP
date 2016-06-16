#ifndef ARRAY3D_H
#define ARRAY3D_H

#include <climits>
#include <vector>

typedef unsigned int uint;

// uncompressed 2D double-precision array (for comparison)
class array3d {
public:
	array3d(uint nx, uint ny, uint nz, uint precision) : nx(nx), ny(ny), nz(nz), data(nx * ny * nz, 0.0) {}
	size_t size() const { return data.size(); }
	double rate() const { return CHAR_BIT * sizeof(double); }
	double& operator()(uint x, uint y, uint z) { return data[x + nx * y + nx * ny * z]; }
	const double& operator()(uint x, uint y, uint z) const { return data[x + nx * y + nx * ny * z]; }
	double& operator[](uint i) { return data[i]; }
	const double& operator[](uint i) const { return data[i]; }
protected:
	uint nx;
	uint ny;
	uint nz;
	std::vector<double> data;
};

#endif
