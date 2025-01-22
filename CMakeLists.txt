cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

if(PLUGIN_SYCL)
  string(REPLACE " -isystem ${CONDA_PREFIX}/include" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

project(xgboost LANGUAGES CXX C VERSION 3.0.0)
include(cmake/Utils.cmake)
list(APPEND CMAKE_MODULE_PATH "${xgboost_SOURCE_DIR}/cmake/modules")

# These policies are already set from 3.18 but we still need to set the policy
# default variables here for lower minimum versions in the submodules
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0076 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0079 NEW)

message(STATUS "CMake version ${CMAKE_VERSION}")

# Check compiler versions
# Use recent compilers to ensure that std::filesystem is available
if(MSVC)
  if(MSVC_VERSION LESS 1920)
    message(FATAL_ERROR "Need Visual Studio 2019 or newer to build XGBoost")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "8.1")
    message(FATAL_ERROR "Need GCC 8.1 or newer to build XGBoost")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "11.0")
    message(FATAL_ERROR "Need Xcode 11.0 (AppleClang 11.0) or newer to build XGBoost")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    message(FATAL_ERROR "Need Clang 9.0 or newer to build XGBoost")
  endif()
endif()

include(${xgboost_SOURCE_DIR}/cmake/PrefetchIntrinsics.cmake)
find_prefetch_intrinsics()
include(${xgboost_SOURCE_DIR}/cmake/Version.cmake)
write_version()
set_default_configuration_release()

#-- Options
include(CMakeDependentOption)

## User options
option(BUILD_C_DOC "Build documentation for C APIs using Doxygen." OFF)
option(USE_OPENMP "Build with OpenMP support." ON)
option(BUILD_STATIC_LIB "Build static library" OFF)
option(BUILD_DEPRECATED_CLI "Build the deprecated command line interface" OFF)
option(FORCE_SHARED_CRT "Build with dynamic CRT on Windows (/MD)" OFF)
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
option(HIDE_CXX_SYMBOLS "Build shared library and hide all C++ symbols" OFF)
option(KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR "Output build artifacts in CMake binary dir" OFF)
## CUDA
option(USE_CUDA  "Build with GPU acceleration" OFF)
option(USE_NCCL  "Build with NCCL to enable distributed GPU support." OFF)
# This is specifically designed for PyPI binary release and should be disabled for most of the cases.
option(USE_DLOPEN_NCCL "Whether to load nccl dynamically." OFF)
option(BUILD_WITH_SHARED_NCCL "Build with shared NCCL library." OFF)

if(USE_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(GPU_COMPUTE_VER "" CACHE STRING
      "Semicolon separated list of compute versions to be built against, e.g. '35;61'")
  else()
    # Clear any cached values from previous runs
    unset(GPU_COMPUTE_VER)
    unset(GPU_COMPUTE_VER CACHE)
  endif()
endif()

# CUDA device LTO was introduced in CMake v3.25 and requires host LTO to also be enabled but can still
# be explicitly disabled allowing for LTO on host only, host and device, or neither, but device-only LTO
# is not a supproted configuration
cmake_dependent_option(USE_CUDA_LTO
  "Enable link-time optimization for CUDA device code"
  "${CMAKE_INTERPROCEDURAL_OPTIMIZATION}"
  "CMAKE_VERSION VERSION_GREATER_EQUAL 3.25;USE_CUDA;CMAKE_INTERPROCEDURAL_OPTIMIZATION"
  OFF)
## Sanitizers
option(USE_SANITIZER "Use santizer flags" OFF)
option(SANITIZER_PATH "Path to sanitizes.")
set(ENABLED_SANITIZERS "address" "leak" CACHE STRING
  "Semicolon separated list of sanitizer names. E.g 'address;leak'. Supported sanitizers are
address, leak, undefined and thread.")
## Plugins
option(PLUGIN_RMM "Build with RAPIDS Memory Manager (RMM)" OFF)
option(PLUGIN_FEDERATED "Build with Federated Learning" OFF)
## TODO: 1. Add check if DPC++ compiler is used for building
option(PLUGIN_SYCL "SYCL plugin" OFF)
option(ADD_PKGCONFIG "Add xgboost.pc into system." ON)

#-- Checks for building XGBoost
if(USE_DEBUG_OUTPUT AND (NOT (CMAKE_BUILD_TYPE MATCHES Debug)))
  message(SEND_ERROR "Do not enable `USE_DEBUG_OUTPUT' with release build.")
endif()
if(USE_NCCL AND NOT (USE_CUDA))
  message(SEND_ERROR "`USE_NCCL` must be enabled with `USE_CUDA` flag.")
endif()
if(USE_DEVICE_DEBUG AND NOT (USE_CUDA))
  message(SEND_ERROR "`USE_DEVICE_DEBUG` must be enabled with `USE_CUDA` flag.")
endif()
if(BUILD_WITH_SHARED_NCCL AND (NOT USE_NCCL))
  message(SEND_ERROR "Build XGBoost with -DUSE_NCCL=ON to enable BUILD_WITH_SHARED_NCCL.")
endif()
if(USE_DLOPEN_NCCL AND (NOT USE_NCCL))
  message(SEND_ERROR "Build XGBoost with -DUSE_NCCL=ON to enable USE_DLOPEN_NCCL.")
endif()
if(USE_DLOPEN_NCCL AND (NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux")))
  message(SEND_ERROR "`USE_DLOPEN_NCCL` supports only Linux at the moment.")
endif()
if(JVM_BINDINGS AND R_LIB)
  message(SEND_ERROR "`R_LIB' is not compatible with `JVM_BINDINGS' as they both have customized configurations.")
endif()
if(R_LIB AND GOOGLE_TEST)
  message(
    WARNING
    "Some C++ tests will fail with `R_LIB` enabled, as R package redirects some functions to R runtime implementation."
  )
endif()
if(PLUGIN_RMM AND NOT (USE_CUDA))
  message(SEND_ERROR "`PLUGIN_RMM` must be enabled with `USE_CUDA` flag.")
endif()
if(PLUGIN_RMM AND NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "Clang") OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")))
  message(SEND_ERROR "`PLUGIN_RMM` must be used with GCC or Clang compiler.")
endif()
if(PLUGIN_RMM AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Linux"))
  message(SEND_ERROR "`PLUGIN_RMM` must be used with Linux.")
endif()
if(ENABLE_ALL_WARNINGS)
  if((NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang") AND (NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
    message(SEND_ERROR "ENABLE_ALL_WARNINGS is only available for Clang and GCC.")
  endif()
endif()
if(BUILD_STATIC_LIB AND (R_LIB OR JVM_BINDINGS))
  message(SEND_ERROR "Cannot build a static library libxgboost.a when R or JVM packages are enabled.")
endif()
if(PLUGIN_FEDERATED)
  if(CMAKE_CROSSCOMPILING)
    message(SEND_ERROR "Cannot cross compile with federated learning support")
  endif()
  if(BUILD_STATIC_LIB)
    message(SEND_ERROR "Cannot build static lib with federated learning support")
  endif()
  if(R_LIB OR JVM_BINDINGS)
    message(SEND_ERROR "Cannot enable federated learning support when R or JVM packages are enabled.")
  endif()
  if(WIN32)
    message(SEND_ERROR "Federated learning not supported for Windows platform")
  endif()
endif()

#-- Removed options
if(USE_AVX)
  message(SEND_ERROR  "The option `USE_AVX` is deprecated as experimental AVX features have been removed from XGBoost.")
endif()
if(PLUGIN_LZ4)
  message(SEND_ERROR  "The option `PLUGIN_LZ4` is removed from XGBoost.")
endif()
if(RABIT_BUILD_MPI)
  message(SEND_ERROR "The option `RABIT_BUILD_MPI` has been removed from XGBoost.")
endif()
if(USE_S3)
  message(SEND_ERROR "The option `USE_S3` has been removed from XGBoost")
endif()
if(USE_AZURE)
  message(SEND_ERROR "The option `USE_AZURE` has been removed from XGBoost")
endif()
if(USE_HDFS)
  message(SEND_ERROR "The option `USE_HDFS` has been removed from XGBoost")
endif()
if(PLUGIN_DENSE_PARSER)
  message(SEND_ERROR "The option `PLUGIN_DENSE_PARSER` has been removed from XGBoost.")
endif()

#-- Sanitizer
if(USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif()

if(USE_CUDA)
  set(USE_OPENMP ON CACHE BOOL "CUDA requires OpenMP" FORCE)
  # `export CXX=' is ignored by CMake CUDA.
  if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND NOT DEFINED ENV{CUDAHOSTCXX})
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH
      "The compiler executable to use when compiling host code for CUDA or HIP language files.")
    mark_as_advanced(CMAKE_CUDA_HOST_COMPILER)
    message(STATUS "Configured CUDA host compiler: ${CMAKE_CUDA_HOST_COMPILER}")
  endif()

  if(NOT DEFINED CMAKE_CUDA_RUNTIME_LIBRARY)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
  endif()

  enable_language(CUDA)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA version must be at least 11.0!")
  endif()
  if(DEFINED GPU_COMPUTE_VER)
    compute_cmake_cuda_archs("${GPU_COMPUTE_VER}")
  endif()

  find_package(CUDAToolkit REQUIRED)
  find_package(CCCL CONFIG)
  if(CCCL_FOUND)
    message(STATUS "Standalone CCCL found.")
  else()
    message(STATUS "Standalone CCCL not found. Attempting to use CCCL from CUDA Toolkit...")
    find_package(CCCL CONFIG
      HINTS ${CUDAToolkit_LIBRARY_DIR}/cmake)
    if(NOT CCCL_FOUND)
      message(STATUS "Could not locate CCCL from CUDA Toolkit. Using Thrust and CUB from CUDA Toolkit...")
      find_package(libcudacxx CONFIG REQUIRED
        HINTS ${CUDAToolkit_LIBRARY_DIR}/cmake)
      find_package(CUB CONFIG REQUIRED
        HINTS ${CUDAToolkit_LIBRARY_DIR}/cmake)
      find_package(Thrust CONFIG REQUIRED
        HINTS ${CUDAToolkit_LIBRARY_DIR}/cmake)
      thrust_create_target(Thrust HOST CPP DEVICE CUDA)
      add_library(CCCL::CCCL INTERFACE IMPORTED GLOBAL)
      target_link_libraries(CCCL::CCCL INTERFACE libcudacxx::libcudacxx CUB::CUB Thrust)
    endif()
  endif()
  # Define guard macros to prevent windows.h from conflicting with winsock2.h
  if(WIN32)
    target_compile_definitions(CCCL::CCCL INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN _WINSOCKAPI_)
  endif()
endif()

if(FORCE_COLORED_OUTPUT AND (CMAKE_GENERATOR STREQUAL "Ninja") AND
    ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
      (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always")
endif()

find_package(Threads REQUIRED)

# -- OpenMP
include(cmake/FindOpenMPMacOS.cmake)
if(USE_OPENMP)
  if(APPLE)
    find_openmp_macos()
  else()
    find_package(OpenMP REQUIRED)
  endif()
endif()

# Add for IBM i
if(${CMAKE_SYSTEM_NAME} MATCHES "OS400")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> -X64 qc <TARGET> <OBJECTS>")
endif()

if(USE_NCCL)
  find_package(Nccl REQUIRED)
endif()

if(MSVC)
  if(FORCE_SHARED_CRT)
    message(STATUS "XGBoost: Using dynamically linked MSVC runtime...")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  else()
    message(STATUS "XGBoost: Using statically linked MSVC runtime...")
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  endif()
endif()

# dmlc-core
set(DMLC_FORCE_SHARED_CRT ${FORCE_SHARED_CRT})
add_subdirectory(${xgboost_SOURCE_DIR}/dmlc-core)

if(MSVC)
  if(TARGET dmlc_unit_tests)
    target_compile_options(
        dmlc_unit_tests PRIVATE
        -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE
    )
  endif()
endif()

# core xgboost
add_subdirectory(${xgboost_SOURCE_DIR}/src)
target_link_libraries(objxgboost PUBLIC dmlc)

# Link -lstdc++fs for GCC 8.x
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
  target_link_libraries(objxgboost PUBLIC stdc++fs)
endif()

# Exports some R specific definitions and objects
if(R_LIB)
  add_subdirectory(${xgboost_SOURCE_DIR}/R-package)
endif()

# This creates its own shared library `xgboost4j'.
if(JVM_BINDINGS)
  add_subdirectory(${xgboost_SOURCE_DIR}/jvm-packages)
endif()

# Plugin
add_subdirectory(${xgboost_SOURCE_DIR}/plugin)

if(PLUGIN_RMM)
  find_package(rmm REQUIRED)

  # Patch the rmm targets so they reference the static cudart
  # Remove this patch once RMM stops specifying cudart requirement
  # (since RMM is a header-only library, it should not specify cudart in its CMake config)
  get_target_property(rmm_link_libs rmm::rmm INTERFACE_LINK_LIBRARIES)
  list(REMOVE_ITEM rmm_link_libs CUDA::cudart)
  list(APPEND rmm_link_libs CUDA::cudart_static)
  set_target_properties(rmm::rmm PROPERTIES INTERFACE_LINK_LIBRARIES "${rmm_link_libs}")
endif()

if(PLUGIN_SYCL)
  set(CMAKE_CXX_LINK_EXECUTABLE
      "icpx <FLAGS> <CMAKE_CXX_LINK_FLAGS> -qopenmp <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>")
  set(CMAKE_CXX_CREATE_SHARED_LIBRARY
      "icpx <CMAKE_SHARED_LIBRARY_CXX_FLAGS> -qopenmp <LANGUAGE_COMPILE_FLAGS> \
      <CMAKE_SHARED_LIBRARY_CREATE_CXX_FLAGS> <SONAME_FLAG>,<TARGET_SONAME> \
      -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

#-- library
if(BUILD_STATIC_LIB)
  add_library(xgboost STATIC)
else()
  add_library(xgboost SHARED)
endif()
target_link_libraries(xgboost PRIVATE objxgboost)
target_include_directories(xgboost
  INTERFACE
  $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>)
#-- End shared library

#-- CLI for xgboost
if(BUILD_DEPRECATED_CLI)
  add_executable(runxgboost ${xgboost_SOURCE_DIR}/src/cli_main.cc)
  target_link_libraries(runxgboost PRIVATE objxgboost)
  target_include_directories(runxgboost
    PRIVATE
    ${xgboost_SOURCE_DIR}/include
    ${xgboost_SOURCE_DIR}/dmlc-core/include
  )
  set_target_properties(runxgboost PROPERTIES OUTPUT_NAME xgboost)
  xgboost_target_properties(runxgboost)
  xgboost_target_link_libraries(runxgboost)
  xgboost_target_defs(runxgboost)

  if(KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR)
    set_output_directory(runxgboost ${xgboost_BINARY_DIR})
  else()
    set_output_directory(runxgboost ${xgboost_SOURCE_DIR})
  endif()
endif()
#-- End CLI for xgboost

# Common setup for all targets
foreach(target xgboost objxgboost dmlc)
  xgboost_target_properties(${target})
  xgboost_target_link_libraries(${target})
  xgboost_target_defs(${target})
endforeach()

if(JVM_BINDINGS)
  xgboost_target_properties(xgboost4j)
  xgboost_target_link_libraries(xgboost4j)
  xgboost_target_defs(xgboost4j)
endif()

if(USE_OPENMP AND APPLE)
  patch_openmp_path_macos(xgboost libxgboost)
endif()

if(KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR)
  set_output_directory(xgboost ${xgboost_BINARY_DIR}/lib)
else()
  set_output_directory(xgboost ${xgboost_SOURCE_DIR}/lib)
endif()

# Ensure these two targets do not build simultaneously, as they produce outputs with conflicting names
if(BUILD_DEPRECATED_CLI)
  add_dependencies(xgboost runxgboost)
endif()

#-- Installing XGBoost
if(R_LIB)
  include(cmake/RPackageInstallTargetSetup.cmake)
  set_target_properties(xgboost PROPERTIES PREFIX "")
  if(APPLE)
    set_target_properties(xgboost PROPERTIES SUFFIX ".so")
  endif()
  setup_rpackage_install_target(xgboost "${CMAKE_CURRENT_BINARY_DIR}/R-package-install")
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dummy_inst")
endif()
if(MINGW)
  set_target_properties(xgboost PROPERTIES PREFIX "")
endif()

if(BUILD_C_DOC)
  include(cmake/Doc.cmake)
  run_doxygen()
endif()

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
if(BUILD_STATIC_LIB)
  if(BUILD_DEPRECATED_CLI)
    set(INSTALL_TARGETS xgboost runxgboost objxgboost dmlc)
  else()
    set(INSTALL_TARGETS xgboost objxgboost dmlc)
  endif()
else()
  if(BUILD_DEPRECATED_CLI)
    set(INSTALL_TARGETS xgboost runxgboost)
  else()
    set(INSTALL_TARGETS xgboost)
  endif()
endif()

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
if(GOOGLE_TEST)
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
    @ONLY
    NEWLINE_STYLE UNIX)
  if(BUILD_DEPRECATED_CLI)
    add_test(
      NAME TestXGBoostCLI
      COMMAND runxgboost ${xgboost_BINARY_DIR}/tests/cli/machine.conf
      WORKING_DIRECTORY ${xgboost_BINARY_DIR})
    set_tests_properties(TestXGBoostCLI
      PROPERTIES
      PASS_REGULAR_EXPRESSION ".*test-rmse:0.087.*")
  endif()
endif()

# Add xgboost.pc
if(ADD_PKGCONFIG)
  configure_file(${xgboost_SOURCE_DIR}/cmake/xgboost.pc.in ${xgboost_BINARY_DIR}/xgboost.pc @ONLY)

  install(
    FILES ${xgboost_BINARY_DIR}/xgboost.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
endif()
