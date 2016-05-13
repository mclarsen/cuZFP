include_directories(${CMAKE_CURRENT_BINARY_DIR}
        "${PROJECT_SOURCE_DIR}/inc"
)

##############################################################################
# Use one executable only.
ADD_EXECUTABLE(main
    main.c
)
TARGET_LINK_LIBRARIES(main
    zfp
)

