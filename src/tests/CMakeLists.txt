###############################################################################
#
#  Testing infrastructure
#
###############################################################################

set(TESTS t_encode_decode
          t_encode_decode_rand 
          t_sanity_check)

foreach(TEST ${TESTS})
  message(STATUS "Adding unit test: ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON cuZFP)
endforeach()
