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
    block2.h 
    decode1f.c  
    decode2f.c  
    decode3f.c  
    encode1f.c  
    encode2f.c  
    encode3f.c  
    traitsd.h  
    zfp.c
    block1.h     
    block3.h  
    decode1d.c      
    decode2d.c  
    decode3d.c  
    encode1d.c  
    encode2d.c  
    encode3d.c  
    traitsf.h

)

IF(WIN32)
		ADD_CUSTOM_COMMAND(
				TARGET zfp
				POST_BUILD
				COMMAND ${CMAKE_COMMAND} -E copy $(ConfigurationName)/zfp.lib ${PROJECT_SOURCE_DIR}/lib/zfp.lib
		)
ELSE(WIN32)
        message(${PROJECT_SOURCE_DIR})
		ADD_CUSTOM_COMMAND(
				TARGET zfp
				POST_BUILD
				#for some reason changes to "source" directory
                                COMMAND ${CMAKE_COMMAND} -E copy libzfp.a ${PROJECT_SOURCE_DIR}/lib/libzfp.a
		)
ENDIF(WIN32)

