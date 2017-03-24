###############################################################################
#
#  file: src/cuZFP/CMakeLists.txt 
#
###############################################################################

set(cuZFP_sources
    encode.cu
    cuZFP.cu #main entry point
    cuZFP.cuh
    decode.cuh
    encode.cuh
    BitStream.cuh
    WriteBitter.cuh
    ErrorCheck.cu)

set(cuZFP_headers
    constant_setup.cuh
    shared.h
    ull128.h
    cuZFP.h
    ErrorCheck.h)

cuda_add_library(cuZFP
                 ${cuZFP_sources}
                 ${cuZFP_headers})


      
