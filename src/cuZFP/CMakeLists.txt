###############################################################################
#
#  file: src/cuZFP/CMakeLists.txt 
#
###############################################################################

set(cuZFP_sources
    encode.cu
    aligned_writer.cuh
    cuZFP.cu #main entry point
    decode.cuh
    decode1.cuh
    decode2.cuh
    encode.cuh
    encode1.cuh
    encode2.cuh
    BitStream.cuh
    type_info.cuh
    WriteBitter.cuh
    ErrorCheck.cu)

set(cuZFP_headers
    constant_setup.cuh
    shared.h
    ull128.h
    cuZFP.h
    zfp_structs.h
    ErrorCheck.h)

cuda_add_library(cuZFP
                 ${cuZFP_sources}
                 ${cuZFP_headers})


      
