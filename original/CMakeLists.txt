CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

PROJECT(ZFPParallel)
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
add_definitions( -DCUDA_4_0_HIGHER=1)


set(CUDA_NVCC_FLAGS "-arch;sm_20;--ptxas-options=-v;-use_fast_math")
#for 650 ti boost
#set(CUDA_NVCC_FLAGS "-arch;sm_30")
#for k20
#set(CUDA_NVCC_FLAGS " -gencode arch=compute_35,code=sm_35;--ptxas-options=-v;-use_fast_math")

INCLUDE_DIRECTORIES(
	"/usr/local/include"
	"/usr/include"
)

LINK_DIRECTORIES(
	"/usr/lib"
	"/usr/local/lib"
    "${PROJECT_SOURCE_DIR}/lib"
)
#set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-pg -march=native -msse2")

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-pg -march=native -msse2")

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_CXX_FLAGS}")


#ADD_DEFINITIONS("-std=c++0x -march=native -msse2")

#SET(CMAKE_BUILD_TYPE Release)
SET(CMAKE_BUILD_TYPE Debug)
SUBDIRS(
    src
    examples
  )
