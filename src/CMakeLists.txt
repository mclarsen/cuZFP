###############################################################################
#
#   CMake lists cuZFP
#
###############################################################################

cmake_minimum_required(VERSION 3.0)

##################################
# cmake policy selections
##################################

cmake_policy(SET CMP0042 OLD)

project(cuZFP)

##################################
# cmake basic setup 
##################################
include(CMake/CMakeBasics.cmake)


##################################
# Setup Cuda 
##################################
include(CMake/SetupCuda.cmake)

##################################
# Setup Tests 
##################################
include(CMake/SetupTests.cmake)

##################################
# Setup Include Dirs project wide 
##################################
include_directories(${PROJECT_SOURCE_DIR}/cuZFP)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty_builtin/zfp-0.5.0/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty_builtin/gtest-1.7.0/include)

##################################
# build thirdparty libs 
##################################
add_subdirectory(thirdparty_builtin)

##################################
# build cuZFP lib 
##################################
add_subdirectory(cuZFP)

##################################
# build tests 
##################################
add_subdirectory(tests)

##################################
# build utils 
##################################
add_subdirectory(utils)

##################################
# Create CMake importable 
# exports for all targets
##################################
#install(EXPORT cuZFP DESTINATION lib/cmake)
