# cuZFP #

cuZFP: a CUDA implementation of ZFP.

This is research code, YMMV. The code itself is not on the master branch, the most recent version is on zfp_new_v2.0 in the v2.0 directory. The include directory in v2.0 contains:

* BitStream.cuh
* cuZFP.cuh
* decode.cuh
* encode.cuh
* shared.h
* ull128.h
* WriteBitter.cuh 

encode.cuh, decode.cuh, and cuZFP.cuh are, for lack of a better term, the API to cuZFP. BitStream.cuh, shared.h, ul128.h and WriteBitter.cuh are are helper files. Note, these are all header files.

The primary functions used are: cuzfp::encode in encode.cuh and cuzfp::decode in decode.cuh. There are basic parallel primitive functions in cuZFP.cuh as well: cuzfp::transform and cuzfp::reduce. This allows for a "thrust"-esque style of programming and made my life a little easier.

### How do I get set up? ###
A good place to get started is in the tests directory, using the test_diffusion.cu. The tests/CMakeLists.txt lists all the tests programs, so commenting out the other ones might be best. test_diffusion has an analytic diffusion solver, a regular CPU diffusion solver, a ZFP diffusion solver, a regular CUDA diffusion solver, and a cuZFP diffusion solver. It's based on ZFP's diffusion test case. 

### How do I do stuff? ###
Following with the diffusion example, first the v0.5 of ZFP needs to be compiled using CMake. Starting from the root ($ROOT) directory of the cuZFP source:

```
cd $ROOT
mkdir build
cd build
mkdir v0.5
cd v0.5
ccmake ../../v0.5
make
```

Assuming the system is Unix-y, ccmake is the curses-based gui front end for cmake. The Cmake for v0.5 is setup in a way that a static lib for zfp is generated and copied to $(ROOT)/v0.5/lib.  

Next, cuZFP is compiled. 

```
cd $ROOT
cd build
mkdir cuzfp
cd cuzfp
ccmake ../../tests
make
```

The build system is setup to default to v2.0 of cuZFP. Don't change that. By default, it's setup to build test_diffusion.

This should generate an executable, test_diffusion in $ROOT/build/cuzfp/

### Other notes ###
Ignore the CMakeLists.txt in the root directory: it doesn't do anything. cuZFP is header files only, so building the tests builds cuZFP.