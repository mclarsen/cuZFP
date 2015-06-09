include_directories(${CMAKE_CURRENT_BINARY_DIR}
        "${PROJECT_SOURCE_DIR}/inc"
)

##############################################################################
# Use one executable only.
CUDA_ADD_EXECUTABLE(diffusion
    diffusion.cpp
)
TARGET_LINK_LIBRARIES(diffusion
    zfp
)
CUDA_ADD_EXECUTABLE(testzfp
    testzfp.cpp
)
TARGET_LINK_LIBRARIES(testzfp
    zfp
)

CUDA_ADD_EXECUTABLE(zfp_c
    zfp.c
)
TARGET_LINK_LIBRARIES(zfp_c
    zfp
)

CUDA_ADD_EXECUTABLE(zfp_cpp
    zfp.cpp
)
TARGET_LINK_LIBRARIES(zfp_cpp
    zfp
)

