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

foreach(TEST ${TESTS})
  message(STATUS "Adding unit test: ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON cuZFP)
endforeach()
