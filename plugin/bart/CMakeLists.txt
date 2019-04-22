set(BART_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/bart.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/gibbs_updater.cc
  PARENT_SCOPE)

if (GOOGLE_TEST)
  set(BART_TESTS_SOURCE
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_gibbs_updater.cc
    ${CMAKE_CURRENT_LIST_DIR}/tests/test_rv.cc
    PARENT_SCOPE)
endif (GOOGLE_TEST)
