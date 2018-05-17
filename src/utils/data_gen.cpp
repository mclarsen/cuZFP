#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>

#include <cuZFP.h>

static void
usage()
{
  fprintf(stderr, "Usage: data_gen <options>\n");
  fprintf(stderr, "General options:\n");
  fprintf(stderr, "Output:\n");
  fprintf(stderr, "  -o <path> : binary output file (\"-\" for stdout)\n");
  fprintf(stderr, "Array type and dimensions:\n");
  fprintf(stderr, "  -t <i32|i64|f32|f64> : integer or floating scalar type (default = f64)\n");
  fprintf(stderr, "  -1 <nx> : dimensions for 1D array a[nx]\n");
  fprintf(stderr, "  -2 <nx> <ny> : dimensions for 2D array a[ny][nx]\n");
  fprintf(stderr, "  -3 <nx> <ny> <nz> : dimensions for 3D array a[nz][ny][nx]\n");
  exit(EXIT_FAILURE);
}

template<typename T>
void gen_1d(int nx, T*& data, double scale = 10.0)
{
  
  data = new T[nx];
  for(int x = 0; x < nx; ++x)
  {
    double v = double (x) * (3.14/180.);
    T val = static_cast<T>(sin(v)*scale);
    data[x] = val;
  }
}

template<typename T>
void gen_braid(int nx, int ny, int nz, T*& data)
{
  
  data = new T[nx*ny*nz];
  double dx = (double) (4.0 * 3.14) / double(nx - 1);
  double dy = (double) (2.0 * 3.14) / double(ny - 1);
  double dz = (double) (3.0 * 3.14) / double(nz - 1);

  size_t index = 0;

  for(int z = 0; z < nz; ++z)
  {
    double cz =  (z * dz) - 1.5 * 3.14;
    for(int y = 0; y < ny; ++y)
    {
      double cy =  (y * dy) - 3.14;
      for(int x = 0; x < nx; ++x)
      {
        double cx =  (x * dx) + (2.0 * 3.14);
        double cv = sin(cx) + sin(cy);
        cv += 2.0 * cos(sqrt( (cx * cx) / 2.0 + cy*cy) / .75);
        cv += 4.0 * cos( cx * cy / 4.0);

        if(z > 1)
        {
          cv += sin(cz) + 1.5 * cos( sqrt(cx * cx + cy *cy + cz * cz) / 0.75);
        }
        T val = static_cast<T>(cv);
        data[index] = val;
        index++;
      } // x
    } // y
  } // z
}

int main(int argc, char* argv[])
{
  cuZFP::zfp_type type = cuZFP::zfp_type_double;
  uint dims = 0;
  uint nx = 0;
  uint ny = 0;
  uint nz = 0;
  char* outpath = 0;
  
  for (int i = 1; i < argc; i++) {
    if (argv[i][0] != '-' || argv[i][2])
      usage();
    switch (argv[i][1]) {
      case '1':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1)
          usage();
        ny = nz = 1;
        dims = 1;
        break;
      case '2':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &ny) != 1)
          usage();
        nz = 1;
        dims = 2;
        break;
      case '3':
        if (++i == argc || sscanf(argv[i], "%u", &nx) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &ny) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &nz) != 1)
          usage();
        dims = 3;
        break;
      case 'o':
        if (++i == argc)
          usage();
        outpath = argv[i];
        break;
      case 't':
        if (++i == argc)
          usage();
        if (!strcmp(argv[i], "i32"))
          type = cuZFP::zfp_type_int32;
        else if (!strcmp(argv[i], "i64"))
          type = cuZFP::zfp_type_int64;
        else if (!strcmp(argv[i], "f32"))
          type = cuZFP::zfp_type_float;
        else if (!strcmp(argv[i], "f64"))
          type = cuZFP::zfp_type_double;
        else
          usage();
        break;
      default:
        usage();
        break;
    }
  }

  void *data = NULL; 

  if(dims == 1)
  {

    if(type == cuZFP::zfp_type_int32)
    {
      int *p_data;
      gen_1d(nx, p_data);
      data = (void*) p_data;
    }
    else if(type == cuZFP::zfp_type_int64)
    {
      long long int *p_data;
      gen_1d(nx, p_data);
      data = (void*) p_data;
    }
    else if(type == cuZFP::zfp_type_float)
    {
      float *p_data;
      gen_1d(nx, p_data);
      data = (void*) p_data;
    }
    else if(type == cuZFP::zfp_type_double)
    {
      double *p_data;
      gen_1d(nx, p_data);
      data = (void*) p_data;
    }
  }

  if(dims == 2 || dims == 3)
  {
    if(dims == 2) nz = 1;

    if(type == cuZFP::zfp_type_int32)
    {
      int *p_data;
      gen_braid(nx, ny, nz, p_data);
      data = (void*) p_data;
    }
    else if(type == cuZFP::zfp_type_int64)
    {
      long long int *p_data;
      gen_braid(nx, ny, nz, p_data);
      data = (void*) p_data;
    }
    else if(type == cuZFP::zfp_type_float)
    {
      float *p_data;
      gen_braid(nx, ny, nz, p_data);
      data = (void*) p_data;
    }
    else if(type == cuZFP::zfp_type_double)
    {
      double *p_data;
      gen_braid(nx, ny, nz, p_data);
      data = (void*) p_data;
    }
  }
      
  // write the data out
  if(outpath) 
  {
    FILE* file = !strcmp(outpath, "-") ? stdout : fopen(outpath, "wb");
    if (!file) 
    {
      fprintf(stderr, "cannot create output file\n");
      return EXIT_FAILURE;
    }
    if (fwrite(data, cuZFP::zfp_type_size(type), nx * ny * nz, file) != nx * ny * nz) 
    {
      fprintf(stderr, "cannot write output file\n");
      return EXIT_FAILURE;
    }
    fclose(file);
  }

  // cleanup
  if(type == cuZFP::zfp_type_int32)
  {
    int *p_data = (int*) data;
    delete[] p_data;
  }
  else if(type == cuZFP::zfp_type_int64)
  {
    long long int *p_data = (long long int*) data;
    delete[] p_data;
  }
  else if(type == cuZFP::zfp_type_float)
  {
    float *p_data = (float *) data;
    delete[] p_data;
  }
  else if(type == cuZFP::zfp_type_double)
  {
    double *p_data = (double*) data;
    delete[] p_data;
  }
}
