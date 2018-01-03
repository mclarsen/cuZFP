
message(STATUS " [*] Adding cuda_zfp util")

add_executable(cuda_zfp cuda_zfp.cpp)

set_target_properties(cuda_zfp PROPERTIES
                      CXX_STANDARD 11
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENTIONS NO)

target_link_libraries(cuda_zfp  cuZFP)

message(STATUS " [*] Adding data_gen util")

add_executable(data_gen data_gen.cpp)

set_target_properties(data_gen PROPERTIES
                      CXX_STANDARD 11
                      CXX_STANDARD_REQUIRED YES
                      CXX_EXTENTIONS NO)

target_link_libraries(data_gen cuZFP)
