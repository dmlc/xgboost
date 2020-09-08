cmake_minimum_required(VERSION 3.3)

if(R_LIB OR MINGW OR WIN32)
  add_library(rabit src/engine_empty.cc src/c_api.cc)
  set(rabit_libs rabit)
  set_target_properties(rabit
    PROPERTIES CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
else()
  find_package(Threads REQUIRED)

  add_library(rabit src/allreduce_base.cc src/allreduce_robust.cc src/engine.cc src/c_api.cc)
  add_library(rabit_mock_static src/allreduce_base.cc src/allreduce_robust.cc src/engine_mock.cc src/c_api.cc)
  add_library(rabit_mock SHARED src/allreduce_base.cc src/allreduce_robust.cc src/engine_mock.cc src/c_api.cc)
  target_link_libraries(rabit Threads::Threads dmlc)
  target_link_libraries(rabit_mock_static Threads::Threads dmlc)
  target_link_libraries(rabit_mock Threads::Threads dmlc)

  set(rabit_libs rabit rabit_mock rabit_mock_static)
  set_target_properties(rabit rabit_mock rabit_mock_static
    PROPERTIES CXX_STANDARD 14
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
endif(R_LIB OR MINGW OR WIN32)

if(RABIT_BUILD_MPI)
  find_package(MPI REQUIRED)
  if (NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "CXX Interface for MPI is required for building MPI backend.")
  endif (NOT MPI_CXX_FOUND)
  add_library(rabit_mpi src/engine_mpi.cc ${MPI_INCLUDE_PATH})
  target_link_libraries(rabit_mpi ${MPI_CXX_LIBRARIES})
  list(APPEND rabit_libs rabit_mpi)
endif()

# place binaries and libraries according to GNU standards
include(GNUInstallDirs)

# we use this to get code coverage
if ((CMAKE_CONFIGURATION_TYPES STREQUAL "Debug") AND (CMAKE_CXX_COMPILER_ID MATCHES GNU))
  foreach(lib ${rabit_libs})
    target_compile_options(${lib}
      -fprofile-arcs
      -ftest-coverage)
    endforeach()
endif((CMAKE_CONFIGURATION_TYPES STREQUAL "Debug") AND (CMAKE_CXX_COMPILER_ID MATCHES GNU))

foreach(lib ${rabit_libs})
  target_include_directories(${lib} PUBLIC
          "$<BUILD_INTERFACE:${xgboost_SOURCE_DIR}/rabit/include>"
          "$<BUILD_INTERFACE:${xgboost_SOURCE_DIR}/dmlc-core/include>")
endforeach()

if (GOOGLE_TEST AND (NOT WIN32))
  enable_testing()

  # rabit mock based integration tests
  list(REMOVE_ITEM rabit_libs "rabit_mock_static") # remove here to avoid installing it
  set(tests lazy_recover local_recover model_recover)

  foreach(test ${tests})
    add_executable(${test} test/${test}.cc)
    target_link_libraries(${test} rabit_mock_static)
    set_target_properties(${test}  PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
    add_test(NAME ${test} COMMAND ${test} WORKING_DIRECTORY ${xgboost_BINARY_DIR})
  endforeach()

  if(RABIT_BUILD_MPI)
    add_executable(speed_test_mpi test/speed_test.cc)
    target_link_libraries(speed_test_mpi rabit_mpi)
    add_test(NAME speed_test_mpi COMMAND speed_test_mpi WORKING_DIRECTORY ${xgboost_BINARY_DIR})
  endif(RABIT_BUILD_MPI)
endif (GOOGLE_TEST AND (NOT WIN32))

set(include_install_dir "include")

# Headers:
install(
    DIRECTORY "include/"
    DESTINATION "${include_install_dir}"
    FILES_MATCHING PATTERN "*.h"
)
