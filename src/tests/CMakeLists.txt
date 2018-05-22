###############################################################################
#
#  Testing infrastructure
#
###############################################################################

set(TESTS 
          t_encode_decode_3
          t_encode_decode_2
          t_encode_decode_1
          #t_encode_decode_rand_3
          t_sanity_check_3
          t_sanity_check_2
          t_sanity_check_1
          )

        set(CUDA_TESTS 
          t_cuda_mem
          )

foreach(TEST ${TESTS})
  message(STATUS "Adding unit test: ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON cuZFP)
endforeach()

foreach(TEST ${CUDA_TESTS})
  message(STATUS "Adding cuda unit test: ${TEST}")
  add_cuda_test(TEST ${TEST} DEPENDS_ON cuZFP)
endforeach()
