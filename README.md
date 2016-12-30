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


### Other notes ###
Ignore the CMakeLists.txt in the root directory: it doesn't do anything. cuZFP is header files only.