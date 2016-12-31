include_directories(${CMAKE_CURRENT_BINARY_DIR}
        "${PROJECT_SOURCE_DIR}/inc"
        "${PROJECT_SOURCE_DIR}/array"
)

##############################################################################
# Use one executable only.
ADD_EXECUTABLE(diffusion
    diffusion.cpp
)
TARGET_LINK_LIBRARIES(diffusion
    zfp
)
ADD_EXECUTABLE(testzfp
    fields.c
    testzfp.cpp
)
TARGET_LINK_LIBRARIES(testzfp
    zfp
)

set(ZFP_C_LIBS zfp)
if(NOT WIN32)
    LIST(APPEND ZFP_C_LIBS m)
endif(NOT WIN32)

message(${ZFP_C_LIBS})
ADD_EXECUTABLE(zfp_c
    zfp.c
)
TARGET_LINK_LIBRARIES(zfp_c
	${ZFP_C_LIBS}
)

ADD_EXECUTABLE(zfp_cpp
    zfp.c
)
TARGET_LINK_LIBRARIES(zfp_cpp
	${ZFP_C_LIBS}
)

