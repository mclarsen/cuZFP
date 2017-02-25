CMAKE_MINIMUM_REQUIRED(VERSION 2.4)

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
