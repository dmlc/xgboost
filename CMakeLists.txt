cmake_minimum_required(VERSION 3.3)
project(xgboost LANGUAGES CXX C VERSION 1.0.0)
include(cmake/Utils.cmake)
list(APPEND CMAKE_MODULE_PATH "${xgboost_SOURCE_DIR}/cmake/modules")
cmake_policy(SET CMP0022 NEW)

if ((${CMAKE_VERSION} VERSION_GREATER 3.13) OR (${CMAKE_VERSION} VERSION_EQUAL 3.13))
  cmake_policy(SET CMP0077 NEW)
endif ((${CMAKE_VERSION} VERSION_GREATER 3.13) OR (${CMAKE_VERSION} VERSION_EQUAL 3.13))

message(STATUS "CMake version ${CMAKE_VERSION}")
if (MSVC)
  cmake_minimum_required(VERSION 3.11)
endif (MSVC)

if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
  message(FATAL_ERROR "GCC version must be at least 5.0!")
endif()

include(${xgboost_SOURCE_DIR}/cmake/FindPrefetchIntrinsics.cmake)
find_prefetch_intrinsics()
include(${xgboost_SOURCE_DIR}/cmake/Version.cmake)
write_version()
set_default_configuration_release()

#-- Options
option(BUILD_C_DOC "Build documentation for C APIs using Doxygen." OFF)
option(USE_OPENMP "Build with OpenMP support." ON)
## Bindings
option(JVM_BINDINGS "Build JVM bindings" OFF)
option(R_LIB "Build shared library for R package" OFF)
## Dev
option(USE_DEBUG_OUTPUT "Dump internal training results like gradients and predictions to stdout.
Should only be used for debugging." OFF)
option(GOOGLE_TEST "Build google tests" OFF)
option(USE_DMLC_GTEST "Use google tests bundled with dmlc-core submodule" OFF)
option(USE_NVTX "Build with cuda profiling annotations. Developers only." OFF)
set(NVTX_HEADER_DIR "" CACHE PATH "Path to the stand-alone nvtx header")
option(RABIT_MOCK "Build rabit with mock" OFF)
## CUDA
option(USE_CUDA  "Build with GPU acceleration" OFF)
option(USE_NCCL  "Build with NCCL to enable distributed GPU support." OFF)
option(BUILD_WITH_SHARED_NCCL "Build with shared NCCL library." OFF)
set(GPU_COMPUTE_VER "" CACHE STRING
  "Semicolon separated list of compute versions to be built against, e.g. '35;61'")
## Copied From dmlc
option(USE_HDFS "Build with HDFS support" OFF)
option(USE_AZURE "Build with AZURE support" OFF)
option(USE_S3 "Build with S3 support" OFF)
## Sanitizers
option(USE_SANITIZER "Use santizer flags" OFF)
option(SANITIZER_PATH "Path to sanitizes.")
set(ENABLED_SANITIZERS "address" "leak" CACHE STRING
  "Semicolon separated list of sanitizer names. E.g 'address;leak'. Supported sanitizers are
address, leak and thread.")
## Plugins
option(PLUGIN_LZ4 "Build lz4 plugin" OFF)
option(PLUGIN_DENSE_PARSER "Build dense parser plugin" OFF)

#-- Checks for building XGBoost
if (USE_DEBUG_OUTPUT AND (NOT (CMAKE_BUILD_TYPE MATCHES Debug)))
  message(SEND_ERROR "Do not enable `USE_DEBUG_OUTPUT' with release build.")
endif (USE_DEBUG_OUTPUT AND (NOT (CMAKE_BUILD_TYPE MATCHES Debug)))
if (USE_NCCL AND NOT (USE_CUDA))
  message(SEND_ERROR "`USE_NCCL` must be enabled with `USE_CUDA` flag.")
endif (USE_NCCL AND NOT (USE_CUDA))
if (BUILD_WITH_SHARED_NCCL AND (NOT USE_NCCL))
  message(SEND_ERROR "Build XGBoost with -DUSE_NCCL=ON to enable BUILD_WITH_SHARED_NCCL.")
endif (BUILD_WITH_SHARED_NCCL AND (NOT USE_NCCL))
if (JVM_BINDINGS AND R_LIB)
  message(SEND_ERROR "`R_LIB' is not compatible with `JVM_BINDINGS' as they both have customized configurations.")
endif (JVM_BINDINGS AND R_LIB)
if (R_LIB AND GOOGLE_TEST)
  message(WARNING "Some C++ unittests will fail with `R_LIB` enabled,
 as R package redirects some functions to R runtime implementation.")
endif (R_LIB AND GOOGLE_TEST)
if (USE_AVX)
  message(SEND_ERROR  "The option 'USE_AVX' is deprecated as experimental AVX features have been removed from XGBoost.")
endif (USE_AVX)

#-- Sanitizer
if (USE_SANITIZER)
  # Older CMake versions have had troubles with Sanitizer
  cmake_minimum_required(VERSION 3.12)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif (USE_SANITIZER)

if (USE_CUDA)
  cmake_minimum_required(VERSION 3.12)
  SET(USE_OPENMP ON CACHE BOOL "CUDA requires OpenMP" FORCE)
  # `export CXX=' is ignored by CMake CUDA.
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  message(STATUS "Configured CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")

  enable_language(CUDA)
  set(GEN_CODE "")
  format_gencode_flags("${GPU_COMPUTE_VER}" GEN_CODE)
  message(STATUS "CUDA GEN_CODE: ${GEN_CODE}")
endif (USE_CUDA)

# dmlc-core
msvc_use_static_runtime()
add_subdirectory(${xgboost_SOURCE_DIR}/dmlc-core)
set_target_properties(dmlc PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)
list(APPEND LINKED_LIBRARIES_PRIVATE dmlc)

# rabit
set(RABIT_BUILD_DMLC OFF)
set(DMLC_ROOT ${xgboost_SOURCE_DIR}/dmlc-core)
set(RABIT_WITH_R_LIB ${R_LIB})
add_subdirectory(rabit)

if (RABIT_MOCK)
  list(APPEND LINKED_LIBRARIES_PRIVATE rabit_mock_static)
else()
  list(APPEND LINKED_LIBRARIES_PRIVATE rabit)
endif(RABIT_MOCK)

# Exports some R specific definitions and objects
if (R_LIB)
  add_subdirectory(${xgboost_SOURCE_DIR}/R-package)
endif (R_LIB)

# core xgboost
add_subdirectory(${xgboost_SOURCE_DIR}/plugin)
add_subdirectory(${xgboost_SOURCE_DIR}/src)
set(XGBOOST_OBJ_SOURCES "${XGBOOST_OBJ_SOURCES};$<TARGET_OBJECTS:objxgboost>")

#-- Shared library
add_library(xgboost SHARED ${XGBOOST_OBJ_SOURCES})
target_include_directories(xgboost
  INTERFACE
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
target_link_libraries(xgboost PRIVATE ${LINKED_LIBRARIES_PRIVATE})

# This creates its own shared library `xgboost4j'.
if (JVM_BINDINGS)
  add_subdirectory(${xgboost_SOURCE_DIR}/jvm-packages)
endif (JVM_BINDINGS)
#-- End shared library

#-- CLI for xgboost
add_executable(runxgboost ${xgboost_SOURCE_DIR}/src/cli_main.cc ${XGBOOST_OBJ_SOURCES})
# For cli_main.cc only
if (USE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_compile_options(runxgboost PRIVATE ${OpenMP_CXX_FLAGS})
endif (USE_OPENMP)

target_include_directories(runxgboost
  PRIVATE
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include
  ${xgboost_SOURCE_DIR}/rabit/include)
target_link_libraries(runxgboost PRIVATE ${LINKED_LIBRARIES_PRIVATE})
set_target_properties(
  runxgboost PROPERTIES
  OUTPUT_NAME xgboost
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON)
#-- End CLI for xgboost

set_output_directory(runxgboost ${xgboost_SOURCE_DIR})
set_output_directory(xgboost ${xgboost_SOURCE_DIR}/lib)
# Ensure these two targets do not build simultaneously, as they produce outputs with conflicting names
add_dependencies(xgboost runxgboost)

#-- Installing XGBoost
if (R_LIB)
  set_target_properties(xgboost PROPERTIES PREFIX "")
  if (APPLE)
    set_target_properties(xgboost PROPERTIES SUFFIX ".so")
  endif (APPLE)
  setup_rpackage_install_target(xgboost ${CMAKE_CURRENT_BINARY_DIR})
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dummy_inst")
endif (R_LIB)
if (MINGW)
  set_target_properties(xgboost PROPERTIES PREFIX "")
endif (MINGW)

if (BUILD_C_DOC)
  include(cmake/Doc.cmake)
  run_doxygen()
endif (BUILD_C_DOC)

include(GNUInstallDirs)
# Install all headers.  Please note that currently the C++ headers does not form an "API".
install(DIRECTORY ${xgboost_SOURCE_DIR}/include/xgboost
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS xgboost runxgboost
  EXPORT XGBoostTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${LIBLEGACY_INCLUDE_DIRS})
install(EXPORT XGBoostTargets
  FILE XGBoostTargets.cmake
  NAMESPACE xgboost::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xgboost)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/xgboost-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/xgboost-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xgboost)
write_basic_package_version_file(
  ${CMAKE_BINARY_DIR}/cmake/xgboost-config-version.cmake
  VERSION ${XGBOOST_VERSION}
  COMPATIBILITY AnyNewerVersion)
install(
  FILES
  ${CMAKE_BINARY_DIR}/cmake/xgboost-config.cmake
  ${CMAKE_BINARY_DIR}/cmake/xgboost-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xgboost)

#-- Test
if (GOOGLE_TEST)
  enable_testing()
  # Unittests.
  add_subdirectory(${xgboost_SOURCE_DIR}/tests/cpp)
  add_test(
    NAME TestXGBoostLib
    COMMAND testxgboost
    WORKING_DIRECTORY ${xgboost_BINARY_DIR})

  # CLI tests
  configure_file(
    ${xgboost_SOURCE_DIR}/tests/cli/machine.conf.in
    ${xgboost_BINARY_DIR}/tests/cli/machine.conf
    @ONLY)
  add_test(
    NAME TestXGBoostCLI
    COMMAND runxgboost ${xgboost_BINARY_DIR}/tests/cli/machine.conf
    WORKING_DIRECTORY ${xgboost_BINARY_DIR})
  set_tests_properties(TestXGBoostCLI
    PROPERTIES
    PASS_REGULAR_EXPRESSION ".*test-rmse:0.087.*")
endif (GOOGLE_TEST)

# For MSVC: Call msvc_use_static_runtime() once again to completely
# replace /MD with /MT. See https://github.com/dmlc/xgboost/issues/4462
# for issues caused by mixing of /MD and /MT flags
msvc_use_static_runtime()
