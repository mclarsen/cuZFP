#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>

#include <cuZFP.h>
/*
File I/O is done using the following combinations of i, o, s, and z:
- i   : read uncompressed
- z   : read compressed
- i, s: read uncompressed, print stats
- i, o: read and write uncompressed
- i, z: read uncompressed, write compressed
- z, o: read compressed, write uncompressed

The 7 major tasks to be accomplished are:
- read uncompressed:  i
- read compressed:    !i
- compress:           i
- write compressed:   i && z
- decompress:         o || s || (!i && z)
- write uncompressed: o
- compute stats:      s
*/
#include <stdint.h>
typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;
typedef uint32_t uint;
typedef int64_t int64;
typedef uint64_t uint64;
typedef unsigned char uchar;
#define ZFP_MIN_BITS     0 /* minimum number of bits per block */
#define ZFP_MAX_BITS  4171 /* maximum number of bits per block */
#define ZFP_MAX_PREC    64 /* maximum precision supported */
#define ZFP_MIN_EXP  -1074 /* minimum floating-point base-2 exponent */

static void
usage()
{
  //fprintf(stderr, "%s\n", zfp_version_string);
  fprintf(stderr, "Usage: zfp <options>\n");
  fprintf(stderr, "General options:\n");
  fprintf(stderr, "  -h : read/write array and compression parameters from/to compressed header\n");
  fprintf(stderr, "  -q : quiet mode; suppress output\n");
  fprintf(stderr, "  -s : print error statistics\n");
  fprintf(stderr, "Input and output:\n");
  fprintf(stderr, "  -i <path> : uncompressed binary input file (\"-\" for stdin)\n");
  fprintf(stderr, "  -o <path> : decompressed binary output file (\"-\" for stdout)\n");
  fprintf(stderr, "  -z <path> : compressed input (w/o -i) or output file (\"-\" for stdin/stdout)\n");
  fprintf(stderr, "Array type and dimensions (needed with -i):\n");
  fprintf(stderr, "  -f : single precision (float type)\n");
  fprintf(stderr, "  -d : double precision (double type)\n");
  fprintf(stderr, "  -t <i32|i64|f32|f64> : integer or floating scalar type\n");
  fprintf(stderr, "  -1 <nx> : dimensions for 1D array a[nx]\n");
  fprintf(stderr, "  -2 <nx> <ny> : dimensions for 2D array a[ny][nx]\n");
  fprintf(stderr, "  -3 <nx> <ny> <nz> : dimensions for 3D array a[nz][ny][nx]\n");
  fprintf(stderr, "Compression parameters (needed with -i):\n");
  fprintf(stderr, "  -r <rate> : fixed rate (# compressed bits per floating-point value)\n");
  fprintf(stderr, "  -p <precision> : fixed precision (# uncompressed bits per value)\n");
  fprintf(stderr, "  -a <tolerance> : fixed accuracy (absolute error tolerance)\n");
  fprintf(stderr, "  -c <minbits> <maxbits> <maxprec> <minexp> : advanced usage\n");
  fprintf(stderr, "      minbits : min # bits per 4^d values in d dimensions\n");
  fprintf(stderr, "      maxbits : max # bits per 4^d values in d dimensions (0 for unlimited)\n");
  fprintf(stderr, "      maxprec : max # bits of precision per value (0 for full)\n");
  fprintf(stderr, "      minexp : min bit plane # coded (-1074 for all bit planes)\n");
  fprintf(stderr, "Examples:\n");
  fprintf(stderr, "  -i file : read uncompressed file and compress to memory\n");
  fprintf(stderr, "  -z file : read compressed file and decompress to memory\n");
  fprintf(stderr, "  -i ifile -z zfile : read uncompressed ifile, write compressed zfile\n");
  fprintf(stderr, "  -z zfile -o ofile : read compressed zfile, write decompressed ofile\n");
  fprintf(stderr, "  -i ifile -o ofile : read ifile, compress, decompress, write ofile\n");
  fprintf(stderr, "  -i file -s : read uncompressed file, compress to memory, print stats\n");
  fprintf(stderr, "  -i - -o - -s : read stdin, compress, decompress, write stdout, print stats\n");
  fprintf(stderr, "  -f -3 100 100 100 -r 16 : 2x fixed-rate compression of 100x100x100 floats\n");
  fprintf(stderr, "  -d -1 1000000 -r 32 : 2x fixed-rate compression of 1M doubles\n");
  fprintf(stderr, "  -d -2 1000 1000 -p 32 : 32-bit precision compression of 1000x1000 doubles\n");
  fprintf(stderr, "  -d -1 1000000 -a 1e-9 : compression of 1M doubles with < 1e-9 max error\n");
  fprintf(stderr, "  -d -1 1000000 -c 64 64 0 -1074 : 4x fixed-rate compression of 1M doubles\n");
  exit(EXIT_FAILURE);
}


int main(int argc, char* argv[])
{
  /* default settings */
  //zfp_type type = zfp_type_none;
  cuZFP::ValueType type = cuZFP::none_type;
  size_t typesize = 0;
  uint dims = 0;
  uint nx = 0;
  uint ny = 0;
  uint nz = 0;
  double rate = 0;
  uint precision = 0;
  double tolerance = 0;
  uint minbits = ZFP_MIN_BITS;
  uint maxbits = ZFP_MAX_BITS;
  uint maxprec = ZFP_MAX_PREC;
  int minexp = ZFP_MIN_EXP;
  int header = 0;
  int quiet = 0;
  int stats = 0;
  char* inpath = 0;
  char* zfppath = 0;
  char* outpath = 0;
  char mode = 0;

  /* local variables */
  int i;
  //zfp_field* field = NULL;
  //zfp_stream* zfp = NULL;
  //bitstream* stream = NULL;
  void* fi = NULL;
  void* fo = NULL;
  void* buffer = NULL;
  size_t rawsize = 0;
  size_t zfpsize = 0;
  size_t bufsize = 0;

  if (argc == 1)
    usage();

  /* parse command-line arguments */
  for (i = 1; i < argc; i++) {
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
      case 'a':
        if (++i == argc || sscanf(argv[i], "%lf", &tolerance) != 1)
          usage();
        mode = 'a';
        break;
      case 'c':
        if (++i == argc || sscanf(argv[i], "%u", &minbits) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &maxbits) != 1 ||
            ++i == argc || sscanf(argv[i], "%u", &maxprec) != 1 ||
            ++i == argc || sscanf(argv[i], "%d", &minexp) != 1)
          usage();
        mode = 'c';
        break;
      case 'd':
        type = cuZFP::f64;
        break;
      case 'f':
        type = cuZFP::f32;
        break;
      case 'h':
        header = 1;
        break;
      case 'i':
        if (++i == argc)
          usage();
        inpath = argv[i];
        break;
      case 'o':
        if (++i == argc)
          usage();
        outpath = argv[i];
        break;
      case 'p':
        if (++i == argc || sscanf(argv[i], "%u", &precision) != 1)
          usage();
        mode = 'p';
        break;
      case 'q':
        quiet = 1;
        break;
      case 'r':
        if (++i == argc || sscanf(argv[i], "%lf", &rate) != 1)
          usage();
        mode = 'r';
        break;
      case 's':
        stats = 1;
        break;
      case 't':
        if (++i == argc)
          usage();
        if (!strcmp(argv[i], "i32"))
          type = cuZFP::i32;
        else if (!strcmp(argv[i], "i64"))
          type = cuZFP::i64;
        else if (!strcmp(argv[i], "f32"))
          type = cuZFP::f32;
        else if (!strcmp(argv[i], "f64"))
          type = cuZFP::f64;
        else
          usage();
        break;
      case 'z':
        if (++i == argc)
          usage();
        zfppath = argv[i];
        break;
      default:
        usage();
        break;
    }
  }

  typesize = cuZFP::type_size(type);
  print_type(type);
  /* make sure we have an input file */
  if (!inpath && !zfppath) {
    fprintf(stderr, "must specify uncompressed or compressed input file via -i or -z\n");
    return EXIT_FAILURE;
  }

  /* make sure we know floating-point type */
  if ((inpath || !header) && !typesize) {
    fprintf(stderr, "must specify scalar type via -f, -d, or -t or header via -h\n");
    return EXIT_FAILURE;
  }

  /* make sure we know array dimensions */
  if ((inpath || !header) && !dims) {
    fprintf(stderr, "must specify array dimensions via -1, -2, or -3 or header via -h\n");
    return EXIT_FAILURE;
  }

  /* make sure we know (de)compression mode and parameters */
  if ((inpath || !header) && !mode) {
    fprintf(stderr, "must specify compression parameters via -a, -c, -p, or -r or header via -h\n");
    return EXIT_FAILURE;
  }

  /* make sure we have input file for stats */
  if (stats && !inpath) {
    fprintf(stderr, "must specify input file via -i to compute stats\n");
    return EXIT_FAILURE;
  }

  /* make sure meta data comes from header or command line, not both */
  if (!inpath && zfppath && header && (typesize || dims)) {
    fprintf(stderr, "cannot specify both field type/size and header\n");
    return EXIT_FAILURE;
  }

  //zfp = zfp_stream_open(NULL);
  //field = zfp_field_alloc();

  /* read uncompressed or compressed file */
  if (inpath) {
    /* read uncompressed input file */
    FILE* file = !strcmp(inpath, "-") ? stdin : fopen(inpath, "rb");
    if (!file) {
      fprintf(stderr, "cannot open input file\n");
      return EXIT_FAILURE;
    }
    rawsize = typesize * nx * ny * nz;
    fi = malloc(rawsize);
    if (!fi) {
      fprintf(stderr, "cannot allocate memory\n");
      return EXIT_FAILURE;
    }
    if (fread(fi, typesize, nx * ny * nz, file) != nx * ny * nz) {
      fprintf(stderr, "cannot read input file\n");
      return EXIT_FAILURE;
    }
    fclose(file);
    //zfp_field_set_pointer(field, fi);
  }
  else {
    /* read compressed input file in increasingly large chunks */
    FILE* file = !strcmp(zfppath, "-") ? stdin : fopen(zfppath, "rb");
    if (!file) {
      fprintf(stderr, "cannot open compressed file\n");
      return EXIT_FAILURE;
    }
    bufsize = 0x100;
    do {
      bufsize *= 2;
      buffer = realloc(buffer, bufsize);
      if (!buffer) {
        fprintf(stderr, "cannot allocate memory\n");
        return EXIT_FAILURE;
      }
      zfpsize += fread((uchar*)buffer + zfpsize, 1, bufsize - zfpsize, file);
    } while (zfpsize == bufsize);
    if (ferror(file)) {
      fprintf(stderr, "cannot read compressed file\n");
      return EXIT_FAILURE;
    }
    fclose(file);

    /* associate bit stream with buffer */
    // stream = stream_open(buffer, bufsize);
    //if (!stream) {
    //  fprintf(stderr, "cannot open compressed stream\n");
    //  return EXIT_FAILURE;
    //}
    //zfp_stream_set_bit_stream(zfp, stream);
  }
  cuZFP::cu_zfp compressor;
  /* set field dimensions and (de)compression parameters */
  if (inpath || !header) {
    printf("set fields dims for (de)comp\n");
    if(dims == 1)
    {
      compressor.set_field_size_1d(nx);
    }
    else if(dims == 2)
    {
      compressor.set_field_size_2d(nx, ny);
    }
    else if(dims == 3)
    {
      compressor.set_field_size_3d(nx, ny, nz);
    }
    
    if(mode != 'r')
    {
      printf("Currently, only the fixed rate '-r' mode is supported with CUDA\n");
      return EXIT_FAILURE;
    }
    compressor.set_rate(rate);
  }

  /* compress input file if provided */
  if (inpath) {
    printf("compress input file if provided\n");
    compressor.set_field(fi, type);
    compressor.compress();
    zfpsize = compressor.get_stream_bytes();
    buffer = compressor.get_stream();
    /* optionally write compressed data */
    if (zfppath) {
      FILE* file = !strcmp(zfppath, "-") ? stdout : fopen(zfppath, "wb");
      if (!file) {
        fprintf(stderr, "cannot create compressed file\n");
        return EXIT_FAILURE;
      }
      if (fwrite(buffer, 1, zfpsize, file) != zfpsize) {
        fprintf(stderr, "cannot write compressed file\n");
        return EXIT_FAILURE;
      }
      fclose(file);
    }
  }
  
  /* decompress data if necessary */
  if ((!inpath && zfppath) || outpath || stats) {


    /* decompress data */
    printf("Decompressing\n");
    compressor.set_stream((Word*)buffer, zfpsize, type);
    compressor.decompress();
    fo = compressor.get_field();
    /* optionally write reconstructed data */
    if (outpath) {
      FILE* file = !strcmp(outpath, "-") ? stdout : fopen(outpath, "wb");
      if (!file) {
        fprintf(stderr, "cannot create output file\n");
        return EXIT_FAILURE;
      }
      if (fwrite(fo, typesize, nx * ny * nz, file) != nx * ny * nz) {
        fprintf(stderr, "cannot write output file\n");
        return EXIT_FAILURE;
      }
      fclose(file);
      fo = NULL; // currently deallocated by cuZFP
    }
  }

  /* print compression and error statistics */
  //TODO
  
  free(buffer);
  free(fi);
  free(fo);
  return EXIT_SUCCESS;
}
