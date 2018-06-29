###############################################################################
#
#  file: src/cuZFP/CMakeLists.txt 
#
###############################################################################

set(cuZFP_sources
    cuZFP.cu         # main entry point
    decode1.cuh
    decode2.cuh
    decode3.cuh
    encode1.cuh
    encode2.cuh
    encode3.cuh
    pointers.cuh
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

install(TARGETS cuZFP DESTINATION lib64)
install(FILES cuZFP.h zfp_structs.h DESTINATION include)
