cmake_minimum_required(VERSION 3.3)

project(rabit VERSION 0.3.0 LANGUAGES CXX)

option(RABIT_BUILD_TESTS "Build rabit tests" OFF)
option(RABIT_BUILD_MPI "Build MPI" OFF)
option(RABIT_BUILD_DMLC "Include DMLC_CORE in build" OFF)
option(RABIT_WITH_R_LIB "Fit the strict environment of R" OFF)

option(DMLC_ROOT "Specify root of external dmlc core.")
# by default point to xgboost/dmlc-core
set(DMLC_ROOT ${CMAKE_CURRENT_LIST_DIR}/../dmlc-core)

# moved from xgboost build
if(R_LIB OR MINGW OR WIN32)
  add_library(rabit src/engine_empty.cc src/c_api.cc)
  set(rabit_libs rabit)
  set_target_properties(rabit
          PROPERTIES CXX_STANDARD 11
          CXX_STANDARD_REQUIRED ON
          POSITION_INDEPENDENT_CODE ON)
else()
  find_package(Threads REQUIRED)
  add_library(rabit_empty src/engine_empty.cc src/c_api.cc)
  add_library(rabit_base src/allreduce_base.cc src/engine_base.cc src/c_api.cc)

  add_library(rabit src/allreduce_base.cc src/allreduce_robust.cc src/engine.cc src/c_api.cc)
  add_library(rabit_mock_static src/allreduce_base.cc src/allreduce_robust.cc src/engine_mock.cc src/c_api.cc)
  add_library(rabit_mock SHARED src/allreduce_base.cc src/allreduce_robust.cc src/engine_mock.cc src/c_api.cc)
  target_link_libraries(rabit Threads::Threads)
  target_link_libraries(rabit_mock_static Threads::Threads)
  target_link_libraries(rabit_mock Threads::Threads)

  set(rabit_libs rabit rabit_base rabit_empty rabit_mock rabit_mock_static)
  set_target_properties(rabit rabit_base rabit_empty rabit_mock rabit_mock_static
    PROPERTIES CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
ENDIF(R_LIB OR MINGW OR WIN32)

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
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR})

# we use this to get code coverage
if ((CMAKE_CONFIGURATION_TYPES STREQUAL "Debug") AND (CMAKE_CXX_COMPILER_ID MATCHES GNU))
  foreach(lib ${rabit_libs})
    target_compile_options(${lib}
      -fprofile-arcs
      -ftest-coverage)
    endforeach()
endif((CMAKE_CONFIGURATION_TYPES STREQUAL "Debug") AND (CMAKE_CXX_COMPILER_ID MATCHES GNU))

if(RABIT_BUILD_DMLC)
  set(DMLC_ROOT ${CMAKE_CURRENT_LIST_DIR}/dmlc-core)
endif()

if(DMLC_ROOT)
  message("DMLC_ROOT point to " ${DMLC_ROOT})
endif(DMLC_ROOT)

foreach(lib ${rabit_libs})
  target_include_directories(${lib} PUBLIC
          "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
          "$<BUILD_INTERFACE:${DMLC_ROOT}/include>")
endforeach()

if (RABIT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${rabit_SOURCE_DIR}/test/cpp)

  # rabit mock based integration tests
  list(REMOVE_ITEM rabit_libs "rabit_mock_static") # remove here to avoid installing it
  set(tests lazy_recover local_recover model_recover)

  foreach(test ${tests})
    add_executable(${test} test/${test}.cc)
    target_link_libraries(${test} rabit_mock_static)
    set_target_properties(${test}  PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)
    install(TARGETS ${test} DESTINATION test) # Why are we installing these??
  endforeach()

  if(RABIT_BUILD_MPI)
    add_executable(speed_test_mpi test/speed_test.cc)
    target_link_libraries(speed_test_mpi rabit_mpi)
    install(TARGETS speed_test_mpi DESTINATION test)
  endif()
endif (RABIT_BUILD_TESTS)

# Installation (https://github.com/forexample/package-example) {

# Layout. This works for all platforms:
#   * <prefix>/lib/cmake/<PROJECT-NAME>
#   * <prefix>/lib/
#   * <prefix>/include/
set(CMAKE_INSTALL_PREFIX "${rabit_SOURCE_DIR}")
set(config_install_dir "lib/cmake/${PROJECT_NAME}")
set(include_install_dir "include")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Use:
#   * PROJECT_VERSION
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * TARGETS_EXPORT_NAME
#   * PROJECT_NAME
configure_package_config_file(
    "cmake/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
)

# Targets:
#   * <prefix>/lib/librabit.a
#   * <prefix>/lib/librabit_base
#   * <prefix>/lib/librabit_empty
#   * header location after install: <prefix>/include/rabit/rabit.h
#   * headers can be included by C++ code `#include <rabit/rabit.h>`
install(
    TARGETS ${rabit_libs}
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
    INCLUDES DESTINATION "${include_install_dir}"
)

# Headers:
install(
    DIRECTORY "include/"
    DESTINATION "${include_install_dir}"
    FILES_MATCHING PATTERN "*.h"
)

# Config
#   * <prefix>/lib/cmake/rabit/rabitConfig.cmake
#   * <prefix>/lib/cmake/rabit/rabitConfigVersion.cmake
install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "${config_install_dir}"
)

# Config
#   * <prefix>/lib/cmake/Foo/FooTargets.cmake
install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "${config_install_dir}"
)
# }
