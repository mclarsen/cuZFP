###############################################################################
#
#  Testing infrastructure
#
###############################################################################

set(TESTS t_encode_decode)

foreach(TEST ${TESTS})
  message(STATUS "Adding unit test: ${TEST}")
  add_cpp_test(TEST ${TEST} DEPENDS_ON cuZFP)
endforeach()
