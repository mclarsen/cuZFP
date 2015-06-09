#http://www.bineteri.com/qtwithcmake
# Don't forget to include output directory, otherwise
# the UI file won't be wrapped!


include_directories(${CMAKE_CURRENT_BINARY_DIR}
        "${PROJECT_SOURCE_DIR}/inc"
)

##############################################################################
# Use one executable only.
CUDA_ADD_LIBRARY(zfp
	libzfp.cpp
	
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

