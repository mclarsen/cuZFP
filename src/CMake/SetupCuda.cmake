################################
# Setup for Cuda support.
################################

# use CUDA_BIN_DIR hint, if set during configure
if(CUDA_BIN_DIR)
    set(ENV{CUDA_BIN_PATH} ${CUDA_BIN_DIR})
endif()

find_package(CUDA REQUIRED)

