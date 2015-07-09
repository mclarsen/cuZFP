#http://www.bineteri.com/qtwithcmake
# Don't forget to include output directory, otherwise
# the UI file won't be wrapped!


include_directories(${CMAKE_CURRENT_BINARY_DIR}
        "${PROJECT_SOURCE_DIR}/inc"
)

##############################################################################

# Use one executable only.
CUDA_ADD_LIBRARY(zfp
bitstream.c
decode1f.c
decode1d.c 
encode1f.c 
encode1d.c 
decode2f.c 
decode2d.c 
encode2f.c 
encode2d.c 
decode3f.c 
decode3d.c 
encode3f.c 
encode3d.c
	
)

IF(WIN32)
		ADD_CUSTOM_COMMAND(
				TARGET zfp
				POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy $(ConfigurationName)/zfp.lib ${PROJECT_SOURCE_DIR}/lib/zfp.lib
		)
ELSE(WIN32)
		ADD_CUSTOM_COMMAND(
				TARGET zfp
				POST_BUILD
				#for some reason changes to "source" directory
                                COMMAND ${CMAKE_COMMAND} -E copy libzfp.a ${PROJECT_SOURCE_DIR}/lib/libzfp.a
		)
ENDIF(WIN32)

