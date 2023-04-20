cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(xgboost LANGUAGES CXX C VERSION 2.0.0)
include(cmake/Utils.cmake)
list(APPEND CMAKE_MODULE_PATH "${xgboost_SOURCE_DIR}/cmake/modules")
cmake_policy(SET CMP0022 NEW)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
cmake_policy(SET CMP0063 NEW)

if ((${CMAKE_VERSION} VERSION_GREATER 3.13) OR (${CMAKE_VERSION} VERSION_EQUAL 3.13))
  cmake_policy(SET CMP0077 NEW)
endif ((${CMAKE_VERSION} VERSION_GREATER 3.13) OR (${CMAKE_VERSION} VERSION_EQUAL 3.13))

message(STATUS "CMake version ${CMAKE_VERSION}")

if (CMAKE_COMPILER_IS_GNUCC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 5.0)
  message(FATAL_ERROR "GCC version must be at least 5.0!")
endif()

include(${xgboost_SOURCE_DIR}/cmake/FindPrefetchIntrinsics.cmake)
find_prefetch_intrinsics()
include(${xgboost_SOURCE_DIR}/cmake/Version.cmake)
write_version()
set_default_configuration_release()

#-- Options
## User options
option(BUILD_C_DOC "Build documentation for C APIs using Doxygen." OFF)
option(USE_OPENMP "Build with OpenMP support." ON)
option(BUILD_STATIC_LIB "Build static library" OFF)
option(FORCE_SHARED_CRT "Build with dynamic CRT on Windows (/MD)" OFF)
option(RABIT_BUILD_MPI "Build MPI" OFF)
## Bindings
option(JVM_BINDINGS "Build JVM bindings" OFF)
option(R_LIB "Build shared library for R package" OFF)
## Dev
option(USE_DEBUG_OUTPUT "Dump internal training results like gradients and predictions to stdout.
Should only be used for debugging." OFF)
option(FORCE_COLORED_OUTPUT "Force colored output from compilers, useful when ninja is used instead of make." OFF)
option(ENABLE_ALL_WARNINGS "Enable all compiler warnings. Only effective for GCC/Clang" OFF)
option(LOG_CAPI_INVOCATION "Log all C API invocations for debugging" OFF)
option(GOOGLE_TEST "Build google tests" OFF)
option(USE_DMLC_GTEST "Use google tests bundled with dmlc-core submodule" OFF)
option(USE_DEVICE_DEBUG "Generate CUDA device debug info." OFF)
option(USE_NVTX "Build with cuda profiling annotations. Developers only." OFF)
set(NVTX_HEADER_DIR "" CACHE PATH "Path to the stand-alone nvtx header")
option(RABIT_MOCK "Build rabit with mock" OFF)
option(HIDE_CXX_SYMBOLS "Build shared library and hide all C++ symbols" OFF)
option(KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR "Output build artifacts in CMake binary dir" OFF)
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
address, leak, undefined and thread.")
## Plugins
option(PLUGIN_DENSE_PARSER "Build dense parser plugin" OFF)
option(PLUGIN_RMM "Build with RAPIDS Memory Manager (RMM)" OFF)
option(PLUGIN_FEDERATED "Build with Federated Learning" OFF)
## TODO: 1. Add check if DPC++ compiler is used for building
option(PLUGIN_UPDATER_ONEAPI "DPC++ updater" OFF)
option(ADD_PKGCONFIG "Add xgboost.pc into system." ON)

#-- Checks for building XGBoost
if (USE_DEBUG_OUTPUT AND (NOT (CMAKE_BUILD_TYPE MATCHES Debug)))
  message(SEND_ERROR "Do not enable `USE_DEBUG_OUTPUT' with release build.")
endif (USE_DEBUG_OUTPUT AND (NOT (CMAKE_BUILD_TYPE MATCHES Debug)))
if (USE_NCCL AND NOT (USE_CUDA))
  message(SEND_ERROR "`USE_NCCL` must be enabled with `USE_CUDA` flag.")
endif (USE_NCCL AND NOT (USE_CUDA))
if (USE_DEVICE_DEBUG AND NOT (USE_CUDA))
  message(SEND_ERROR "`USE_DEVICE_DEBUG` must be enabled with `USE_CUDA` flag.")
endif (USE_DEVICE_DEBUG AND NOT (USE_CUDA))
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
if (PLUGIN_LZ4)
  message(SEND_ERROR  "The option 'PLUGIN_LZ4' is removed from XGBoost.")
endif (PLUGIN_LZ4)
if (PLUGIN_RMM AND NOT (USE_CUDA))
  message(SEND_ERROR "`PLUGIN_RMM` must be enabled with `USE_CUDA` flag.")
endif (PLUGIN_RMM AND NOT (USE_CUDA))
if (PLUGIN_RMM AND NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")))
  message(SEND_ERROR "`PLUGIN_RMM` must be used with GCC or Clang compiler.")
endif (PLUGIN_RMM AND NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")))
if (PLUGIN_RMM AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
  message(SEND_ERROR "`PLUGIN_RMM` must be used with Linux.")
endif (PLUGIN_RMM AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
if (ENABLE_ALL_WARNINGS)
  if ((NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(SEND_ERROR "ENABLE_ALL_WARNINGS is only available for Clang and GCC.")
  endif ((NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
endif (ENABLE_ALL_WARNINGS)
if (BUILD_STATIC_LIB AND (R_LIB OR JVM_BINDINGS))
  message(SEND_ERROR "Cannot build a static library libxgboost.a when R or JVM packages are enabled.")
endif (BUILD_STATIC_LIB AND (R_LIB OR JVM_BINDINGS))
if (PLUGIN_FEDERATED)
  if (CMAKE_CROSSCOMPILING)
    message(SEND_ERROR "Cannot cross compile with federated learning support")
  endif ()
  if (BUILD_STATIC_LIB)
    message(SEND_ERROR "Cannot build static lib with federated learning support")
  endif ()
  if (R_LIB OR JVM_BINDINGS)
    message(SEND_ERROR "Cannot enable federated learning support when R or JVM packages are enabled.")
  endif ()
  if (WIN32)
    message(SEND_ERROR "Federated learning not supported for Windows platform")
  endif ()
endif ()

#-- Sanitizer
if (USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif (USE_SANITIZER)

if (USE_CUDA)
  set(USE_OPENMP ON CACHE BOOL "CUDA requires OpenMP" FORCE)
  # `export CXX=' is ignored by CMake CUDA.
  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  message(STATUS "Configured CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")

  enable_language(CUDA)
  if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA version must be at least 11.0!")
  endif()
  set(GEN_CODE "")
  format_gencode_flags("${GPU_COMPUTE_VER}" GEN_CODE)
  add_subdirectory(${PROJECT_SOURCE_DIR}/gputreeshap)
endif (USE_CUDA)

if (FORCE_COLORED_OUTPUT AND (CMAKE_GENERATOR STREQUAL "Ninja") AND
    ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
      (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

find_package(Threads REQUIRED)

if (USE_OPENMP)
  if (APPLE)
    find_package(OpenMP)
    if (NOT OpenMP_FOUND)
      # Try again with extra path info; required for libomp 15+ from Homebrew
      execute_process(COMMAND brew --prefix libomp
                      OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      set(OpenMP_C_FLAGS
        "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_CXX_FLAGS
        "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
      set(OpenMP_C_LIB_NAMES omp)
      set(OpenMP_CXX_LIB_NAMES omp)
      set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
      find_package(OpenMP REQUIRED)
    endif ()
  else ()
    find_package(OpenMP REQUIRED)
  endif ()
endif (USE_OPENMP)
#Add for IBM i
if (${CMAKE_SYSTEM_NAME} MATCHES "OS400")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -X64 qc <TARGET> <OBJECTS>")
endif()

if (USE_NCCL)
  find_package(Nccl REQUIRED)
endif (USE_NCCL)

# dmlc-core
msvc_use_static_runtime()
if (FORCE_SHARED_CRT)
  set(DMLC_FORCE_SHARED_CRT ON)
endif ()
add_subdirectory(${xgboost_SOURCE_DIR}/dmlc-core)

if (MSVC)
  if (TARGET dmlc_unit_tests)
    target_compile_options(dmlc_unit_tests PRIVATE
                           -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
  endif (TARGET dmlc_unit_tests)
endif (MSVC)

# rabit
add_subdirectory(rabit)
if (RABIT_BUILD_MPI)
  find_package(MPI REQUIRED)
endif (RABIT_BUILD_MPI)

# core xgboost
add_subdirectory(${xgboost_SOURCE_DIR}/src)
target_link_libraries(objxgboost PUBLIC dmlc)

# Exports some R specific definitions and objects
if (R_LIB)
  add_subdirectory(${xgboost_SOURCE_DIR}/R-package)
endif (R_LIB)

# This creates its own shared library `xgboost4j'.
if (JVM_BINDINGS)
  add_subdirectory(${xgboost_SOURCE_DIR}/jvm-packages)
endif (JVM_BINDINGS)

# Plugin
add_subdirectory(${xgboost_SOURCE_DIR}/plugin)

if (PLUGIN_RMM)
  find_package(rmm REQUIRED)
endif (PLUGIN_RMM)

#-- library
if (BUILD_STATIC_LIB)
  add_library(xgboost STATIC)
else (BUILD_STATIC_LIB)
  add_library(xgboost SHARED)
endif (BUILD_STATIC_LIB)
target_link_libraries(xgboost PRIVATE objxgboost)
target_include_directories(xgboost
  INTERFACE
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
#-- End shared library

#-- CLI for xgboost
add_executable(runxgboost ${xgboost_SOURCE_DIR}/src/cli_main.cc)
target_link_libraries(runxgboost PRIVATE objxgboost)
target_include_directories(runxgboost
  PRIVATE
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include
  ${xgboost_SOURCE_DIR}/rabit/include
)
set_target_properties(runxgboost PROPERTIES OUTPUT_NAME xgboost)
#-- End CLI for xgboost

# Common setup for all targets
foreach(target xgboost objxgboost dmlc runxgboost)
  xgboost_target_properties(${target})
  xgboost_target_link_libraries(${target})
  xgboost_target_defs(${target})
endforeach()

if (JVM_BINDINGS)
  xgboost_target_properties(xgboost4j)
  xgboost_target_link_libraries(xgboost4j)
  xgboost_target_defs(xgboost4j)
endif (JVM_BINDINGS)

if (KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR)
  set_output_directory(runxgboost ${xgboost_BINARY_DIR})
  set_output_directory(xgboost ${xgboost_BINARY_DIR}/lib)
else ()
  set_output_directory(runxgboost ${xgboost_SOURCE_DIR})
  set_output_directory(xgboost ${xgboost_SOURCE_DIR}/lib)
endif ()
# Ensure these two targets do not build simultaneously, as they produce outputs with conflicting names
add_dependencies(xgboost runxgboost)

#-- Installing XGBoost
if (R_LIB)
  include(cmake/RPackageInstallTargetSetup.cmake)
  set_target_properties(xgboost PROPERTIES PREFIX "")
  if (APPLE)
    set_target_properties(xgboost PROPERTIES SUFFIX ".so")
  endif (APPLE)
  setup_rpackage_install_target(xgboost "${CMAKE_CURRENT_BINARY_DIR}/R-package-install")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dummy_inst")
endif (R_LIB)
if (MINGW)
  set_target_properties(xgboost PROPERTIES PREFIX "")
endif (MINGW)

if (BUILD_C_DOC)
  include(cmake/Doc.cmake)
  run_doxygen()
endif (BUILD_C_DOC)

include(CPack)

include(GNUInstallDirs)
# Install all headers.  Please note that currently the C++ headers does not form an "API".
install(DIRECTORY ${xgboost_SOURCE_DIR}/include/xgboost
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Install libraries. If `xgboost` is a static lib, specify `objxgboost` also, to avoid the
# following error:
#
#  > install(EXPORT ...) includes target "xgboost" which requires target "objxgboost" that is not
#  > in any export set.
#
# https://github.com/dmlc/xgboost/issues/6085
if (BUILD_STATIC_LIB)
  set(INSTALL_TARGETS xgboost runxgboost objxgboost dmlc)
else (BUILD_STATIC_LIB)
  set(INSTALL_TARGETS xgboost runxgboost)
endif (BUILD_STATIC_LIB)

install(TARGETS ${INSTALL_TARGETS}
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
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/xgboost-config.cmake
  ${CMAKE_BINARY_DIR}/cmake/xgboost-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/xgboost)

#-- Test
if (GOOGLE_TEST)
  enable_testing()
  # Unittests.
  add_executable(testxgboost)
  target_link_libraries(testxgboost PRIVATE objxgboost)
  xgboost_target_properties(testxgboost)
  xgboost_target_link_libraries(testxgboost)
  xgboost_target_defs(testxgboost)

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

# Add xgboost.pc
if (ADD_PKGCONFIG)
  configure_file(${xgboost_SOURCE_DIR}/cmake/xgboost.pc.in ${xgboost_BINARY_DIR}/xgboost.pc @ONLY)

  install(
    FILES ${xgboost_BINARY_DIR}/xgboost.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif (ADD_PKGCONFIG)
