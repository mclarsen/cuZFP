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
# build cuZFP lib 
##################################
add_subdirectory(cuZFP)

##################################
# Create CMake importable 
# exports for all targets
##################################
#install(EXPORT cuZFP DESTINATION lib/cmake)
