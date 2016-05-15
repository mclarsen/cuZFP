CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

PROJECT(ZFPCUDATest)

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_CXX_FLAGS}")
IF(WIN32)
ELSE(WIN32)
	SET(LIBOMP gomp)
ENDIF(WIN32)

FIND_PACKAGE(CUDA)
if (CUDA_SDK_ROOT_DIR MATCHES "CUDA_SDK_ROOT_DIR-NOTFOUND")
        IF(WIN32)
                # Look for the SDK stuff
                find_path(CUDA_SDK_ROOT_DIR C/common/inc/cutil.h
                  "$ENV{NVSDKCUDA_ROOT}"
                  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\NVIDIA Corporation\\Installed Products\\NVIDIA SDK 10\\Compute;InstallDir]"
                "/Developer/GPU\ Computing/C"
                )

                # Keep the CUDA_SDK_ROOT_DIR first in order to be able to override the
                # environment variables.
                set(CUDA_SDK_SEARCH_PATH
                  "${CUDA_SDK_ROOT_DIR}"
                  "${CUDA_TOOLKIT_ROOT_DIR}/local/NVSDK0.2"
                  "${CUDA_TOOLKIT_ROOT_DIR}/NVSDK0.2"
                  "${CUDA_TOOLKIT_ROOT_DIR}/NV_CUDA_SDK"
                  "$ENV{HOME}/NVIDIA_CUDA_SDK"
                  "$ENV{HOME}/NVIDIA_CUDA_SDK_MACOSX"
                  "/Developer/CUDA"
                )

        ELSE(WIN32)
                find_path(CUDA_SDK_ROOT_DIR C/common/inc/cutil.h
                                $ENV{NVSDKCUDA_ROOT}
                                ~/NVIDIA_GPU_Computing_SDK
                )
                SET(CUDA_SDK_SEARCH_PATH
                        "${CUDA_SDK_ROOT_DIR}"
                        "~/NVIDIA_GPU_Computing_SDK"
                )
        ENDIF(WIN32)
 endif (CUDA_SDK_ROOT_DIR MATCHES "CUDA_SDK_ROOT_DIR-NOTFOUND")
INCLUDE_DIRECTORIES(
    ${CUDA_SDK_ROOT_DIR}/common/inc
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    )
if(WIN32)
    SET(CUTIL_LIB_DIRECTORY
        "${CUDA_SDK_ROOT_DIR}/common/lib/"
    )
else(WIN32)
    SET(CUTIL_LIB_DIRECTORY
        "${CUDA_SDK_ROOT_DIR}/C/lib/"
    )
endif(WIN32)
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
#list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")


#list(APPEND CUDA_NVCC_FLAGS "-G")
#list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
#list(APPEND CUDA_NVCC_FLAGS "-Xptxas;-O3")

#set(BaseName "binary" CACHE STRING "BaseName chosen by the user at CMake configure time")
set(BUILD_ZFP_PARALLEL "v1.0" CACHE STRING "Build Version 1.0|2.0")
set_property(CACHE BUILD_ZFP_PARALLEL PROPERTY STRINGS "v1.0" "v1.0;v2.0")

#message(${BUILD_ZFP_PARALLEL})

SET (ZFP_PARALLEL_VERSION "v1.0")
IF(${BUILD_ZFP_PARALLEL} MATCHES "v2.0")
    SET(ZFP_PARALLEL_VERSION "v2.0")
ENDIF()


INCLUDE_DIRECTORIES(
	"/usr/local/include"
	"/usr/include"
    "../${ZFP_PARALLEL_VERSION}/"
	"${CMAKE_CURRENT_SOURCE_DIR}/../v0.5/inc"
	"${CMAKE_CURRENT_SOURCE_DIR}/../v0.5/array"
)

message("${CMAKE_CURRENT_SOURCE_DIR}/../")
LINK_DIRECTORIES(
	"/usr/lib"
	"/usr/local/lib"
        ${CUTIL_LIB_DIRECTORY}
	"${CMAKE_CURRENT_SOURCE_DIR}/../v0.5/lib"
)
#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-pg -march=native -msse2")

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pg -march=native -msse2")

#ADD_DEFINITIONS("-std=c++0x -march=native -msse2")
#ADD_DEFINITIONS("-std=c99")
#SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE Debug)
set(ENC_CUH "../${ZFP_PARALLEL_VERSION}/include/encode.cuh")
set(DEC_CUH "../${ZFP_PARALLEL_VERSION}/include/decode.cuh")
set(BIT_CUH "../${ZFP_PARALLEL_VERSION}/include/BitStream.cuh")
if(false)
CUDA_ADD_EXECUTABLE(zfp_test_exp
ErrorCheck.cpp
test_exp.cu
)
TARGET_LINK_LIBRARIES(zfp_test_exp

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_fixed
ErrorCheck.cpp
${ENC_CUH}
test_fixed_point.cu
)
TARGET_LINK_LIBRARIES(zfp_test_fixed
${LIBOMP}
cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_decorrelate
ErrorCheck.cpp
${ENC_CUH}
test_decorrelate.cu
)
TARGET_LINK_LIBRARIES(zfp_test_decorrelate
${LIBOMP}

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_int2uint
ErrorCheck.cpp
${ENC_CUH}
test_int2uint.cu
)
TARGET_LINK_LIBRARIES(zfp_test_int2uint
${LIBOMP}

cudart
)
CUDA_ADD_EXECUTABLE(zfp_test_bitstream
ErrorCheck.cpp
${ENC_CUH}
${BIT_CUH}
test_bitstream.cu
)
TARGET_LINK_LIBRARIES(zfp_test_bitstream
${LIBOMP}

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_inv_xform
ErrorCheck.cpp
${ENC_CUH}
${DEC_CUH}
${BIT_CUH}
test_inv_xform.cu
)
TARGET_LINK_LIBRARIES(zfp_test_inv_xform
${LIBOMP}

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_decode
ErrorCheck.cpp
${ENC_CUH}
${DEC_CUH}
${BIT_CUH}
test_decode.cu
)
TARGET_LINK_LIBRARIES(zfp_test_decode
${LIBOMP}

cudart
)
endif()
if(false)
CUDA_ADD_EXECUTABLE(zfp_test_all
ErrorCheck.cpp
${ENC_CUH}
${DEC_CUH}
${BIT_CUH}
test_all.cu
)
TARGET_LINK_LIBRARIES(zfp_test_all
${LIBOMP}

cudart
)
endif()


if(false)

CUDA_ADD_EXECUTABLE(zfp_test_ull128
ErrorCheck.cpp
${ENC_CUH}
${BIT_CUH}
test_ull128.cu
)
TARGET_LINK_LIBRARIES(zfp_test_ull128
${LIBOMP}

cudart
)
endif()


CUDA_ADD_EXECUTABLE(zfp_test_parallel
ErrorCheck.cpp
${ENC_CUH}
${DEC_CUH}
${BIT_CUH}
test_parallel.cu
)
TARGET_LINK_LIBRARIES(zfp_test_parallel
${LIBOMP}
zfp
cudart
)
