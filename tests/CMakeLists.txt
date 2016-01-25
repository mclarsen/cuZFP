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
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_30,code=sm_30")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_35,code=sm_35")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_50,code=sm_50")
list(APPEND CUDA_NVCC_FLAGS "-gencode arch=compute_52,code=sm_52")
#list(APPEND CUDA_NVCC_FLAGS "-G")
#list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
#list(APPEND CUDA_NVCC_FLAGS "-Xptxas;-O3")


INCLUDE_DIRECTORIES(
	"/usr/local/include"
	"/usr/include"
)

LINK_DIRECTORIES(
	"/usr/lib"
	"/usr/local/lib"
        ${CUTIL_LIB_DIRECTORY}

)
#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-pg -march=native -msse2")

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pg -march=native -msse2")

#ADD_DEFINITIONS("-std=c++0x -march=native -msse2")
#ADD_DEFINITIONS("-std=c99")
#SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE Debug)


CUDA_ADD_EXECUTABLE(zfp_test_exp
ErrorCheck.cpp
test_exp.cu
)
TARGET_LINK_LIBRARIES(zfp_test_exp

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_fixed
ErrorCheck.cpp
encode.cuh
test_fixed_point.cu
)
TARGET_LINK_LIBRARIES(zfp_test_fixed
${LIBOMP}
cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_decorrelate
ErrorCheck.cpp
encode.cuh
test_decorrelate.cu
)
TARGET_LINK_LIBRARIES(zfp_test_decorrelate
${LIBOMP}

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_int2uint
ErrorCheck.cpp
encode.cuh
test_int2uint.cu
)
TARGET_LINK_LIBRARIES(zfp_test_int2uint
${LIBOMP}

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_bitstream
ErrorCheck.cpp
encode.cuh
BitStream.cuh
test_bitstream.cu
)
TARGET_LINK_LIBRARIES(zfp_test_bitstream
${LIBOMP}

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_inv_xform
ErrorCheck.cpp
encode.cuh
decode.cuh
BitStream.cuh
test_inv_xform.cu
)
TARGET_LINK_LIBRARIES(zfp_test_inv_xform
${LIBOMP}

cudart
)

CUDA_ADD_EXECUTABLE(zfp_test_decode
ErrorCheck.cpp
encode.cuh
decode.cuh
BitStream.cuh
test_decode.cu
)
TARGET_LINK_LIBRARIES(zfp_test_decode
${LIBOMP}

cudart
)
