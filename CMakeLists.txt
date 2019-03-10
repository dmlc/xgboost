cmake_minimum_required (VERSION 3.2)
project(xgboost)
include(cmake/Utils.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
find_package(OpenMP)

set_default_configuration_release()
msvc_use_static_runtime()

# Options
## GPUs
option(USE_CUDA  "Build with GPU acceleration" OFF)
option(USE_NVTX  "Build with cuda profiling annotations. Developers only." OFF)
option(USE_NCCL  "Build with multiple GPUs support" OFF)
set(GPU_COMPUTE_VER "" CACHE STRING
  "Space separated list of compute versions to be built against, e.g. '35 61'")
set(NVTX_HEADER_DIR "" CACHE PATH
  "Path to the stand-alone nvtx header")

## Bindings
option(JVM_BINDINGS "Build JVM bindings" OFF)
option(R_LIB "Build shared library for R package" OFF)

## Devs
option(USE_SANITIZER "Use santizer flags" OFF)
option(SANITIZER_PATH "Path to sanitizes.")
set(ENABLED_SANITIZERS "address" "leak" CACHE STRING
  "Semicolon separated list of sanitizer names. E.g 'address;leak'. Supported sanitizers are
address, leak and thread.")
option(GOOGLE_TEST "Build google tests" OFF)

# Plugins
option(PLUGIN_LZ4 "Build lz4 plugin" OFF)
option(PLUGIN_DENSE_PARSER "Build dense parser plugin" OFF)

# Deprecation warning
if(USE_AVX)
  message(WARNING "The option 'USE_AVX' is deprecated as experimental AVX features have been removed from xgboost.")
endif()

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(OpenMP_CXX_FOUND OR OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(MSVC)
  # Multithreaded compilation
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  # Correct error for GCC 5 and cuda
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
  # Performance
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
endif()
if(WIN32 AND MINGW)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++")
endif()

# Check existence of software pre-fetching
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <xmmintrin.h>
int main() {
  char data = 0;
  const char* address = &data;
  _mm_prefetch(address, _MM_HINT_NTA);
  return 0;
}
" XGBOOST_MM_PREFETCH_PRESENT)
check_cxx_source_compiles("
int main() {
  char data = 0;
  const char* address = &data;
  __builtin_prefetch(address, 0, 0);
  return 0;
}
" XGBOOST_BUILTIN_PREFETCH_PRESENT)

# Sanitizer
if(USE_SANITIZER)
  include(cmake/Sanitizer.cmake)
  enable_sanitizers("${ENABLED_SANITIZERS}")
endif(USE_SANITIZER)

# dmlc-core
add_subdirectory(dmlc-core)
set(LINK_LIBRARIES dmlc rabit)

# enable custom logging
add_definitions(-DDMLC_LOG_CUSTOMIZE=1)

# compiled code customizations for R package
if(R_LIB)
  add_definitions(
    -DXGBOOST_STRICT_R_MODE=1
    -DXGBOOST_CUSTOMIZE_GLOBAL_PRNG=1
    -DDMLC_LOG_BEFORE_THROW=0
    -DDMLC_DISABLE_STDIN=1
    -DDMLC_LOG_CUSTOMIZE=1
    -DRABIT_CUSTOMIZE_MSG_
    -DRABIT_STRICT_CXX98_
  )
endif()

# Gather source files
include_directories (
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/rabit/include
)

# Generate configurable header
set(CMAKE_LOCAL "${PROJECT_SOURCE_DIR}/cmake")
set(INCLUDE_ROOT "${PROJECT_SOURCE_DIR}/include")
message(STATUS "${CMAKE_LOCAL}/build_config.h.in -> ${INCLUDE_ROOT}/xgboost/build_config.h")
configure_file("${CMAKE_LOCAL}/build_config.h.in" "${INCLUDE_ROOT}/xgboost/build_config.h")

file(GLOB_RECURSE SOURCES
    src/*.cc
    src/*.h
    include/*.h
)

# Only add main function for executable target
list(REMOVE_ITEM SOURCES ${PROJECT_SOURCE_DIR}/src/cli_main.cc)

file(GLOB_RECURSE CUDA_SOURCES
    src/*.cu
    src/*.cuh
)

# Add plugins to source files
if(PLUGIN_LZ4)
  list(APPEND SOURCES plugin/lz4/sparse_page_lz4_format.cc)
  link_libraries(lz4)
endif()
if(PLUGIN_DENSE_PARSER)
  list(APPEND SOURCES plugin/dense_parser/dense_libsvm.cc)
endif()

# rabit
# TODO: Use CMakeLists.txt from rabit.
set(RABIT_SOURCES
    rabit/src/allreduce_base.cc
    rabit/src/allreduce_robust.cc
    rabit/src/engine.cc
    rabit/src/c_api.cc
)
set(RABIT_EMPTY_SOURCES
    rabit/src/engine_empty.cc
    rabit/src/c_api.cc
)

if(MINGW OR R_LIB)
  # build a dummy rabit library
  add_library(rabit STATIC ${RABIT_EMPTY_SOURCES})
else()
  add_library(rabit STATIC ${RABIT_SOURCES})
endif()

if (GENERATE_COMPILATION_DATABASE)
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif (GENERATE_COMPILATION_DATABASE)

if(USE_CUDA AND (NOT GENERATE_COMPILATION_DATABASE))
  find_package(CUDA 8.0 REQUIRED)
  cmake_minimum_required(VERSION 3.5)

  add_definitions(-DXGBOOST_USE_CUDA)

  include_directories(cub)

  if(USE_NCCL)
    find_package(Nccl REQUIRED)
    cuda_include_directories(${NCCL_INCLUDE_DIR})
    add_definitions(-DXGBOOST_USE_NCCL)
  endif()

  if(USE_NVTX)
    cuda_include_directories("${NVTX_HEADER_DIR}")
    add_definitions(-DXGBOOST_USE_NVTX)
  endif()

  set(GENCODE_FLAGS "")
  format_gencode_flags("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
  message("cuda architecture flags: ${GENCODE_FLAGS}")

  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;--expt-relaxed-constexpr;${GENCODE_FLAGS};-lineinfo;")
  if(NOT MSVC)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC; -Xcompiler -Werror; -std=c++11")
  endif()

  cuda_add_library(gpuxgboost ${CUDA_SOURCES} STATIC)

  if(USE_NCCL)
    link_directories(${NCCL_LIBRARY})
    target_link_libraries(gpuxgboost ${NCCL_LIB_NAME})
  endif()

  list(APPEND LINK_LIBRARIES gpuxgboost)

elseif (USE_CUDA AND GENERATE_COMPILATION_DATABASE)
  # Enable CUDA language to generate a compilation database.
  cmake_minimum_required(VERSION 3.8)

  find_package(CUDA 8.0 REQUIRED)
  enable_language(CUDA)
  set(CMAKE_CUDA_COMPILER clang++)
  set(CUDA_SEPARABLE_COMPILATION ON)
  if (NOT CLANG_CUDA_GENCODE)
    set(CLANG_CUDA_GENCODE "--cuda-gpu-arch=sm_35")
  endif (NOT CLANG_CUDA_GENCODE)
  set(CMAKE_CUDA_FLAGS " -Wno-deprecated  ${CLANG_CUDA_GENCODE} -fPIC ${GENCODE} -std=c++11 -x cuda")
  message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")

  add_library(gpuxgboost STATIC ${CUDA_SOURCES})

  if(USE_NCCL)
    find_package(Nccl REQUIRED)
    target_include_directories(gpuxgboost PUBLIC ${NCCL_INCLUDE_DIR})
    target_compile_definitions(gpuxgboost PUBLIC -DXGBOOST_USE_NCCL)
    target_link_libraries(gpuxgboost PUBLIC ${NCCL_LIB_NAME})
  endif()

  target_compile_definitions(gpuxgboost PUBLIC -DXGBOOST_USE_CUDA)
  # A hack for CMake to make arguments valid for clang++
  string(REPLACE "-x cu" "-x cuda" CMAKE_CUDA_COMPILE_PTX_COMPILATION
    ${CMAKE_CUDA_COMPILE_PTX_COMPILATION})
  string(REPLACE "-x cu" "-x cuda" CMAKE_CUDA_COMPILE_WHOLE_COMPILATION
    ${CMAKE_CUDA_COMPILE_WHOLE_COMPILATION})
  string(REPLACE "-x cu" "-x cuda" CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION
    ${CMAKE_CUDA_COMPILE_SEPARABLE_COMPILATION})
  target_include_directories(gpuxgboost PUBLIC cub)
endif()


# flags and sources for R-package
if(R_LIB)
  file(GLOB_RECURSE R_SOURCES
    R-package/src/*.h
    R-package/src/*.c
    R-package/src/*.cc
  )
  list(APPEND SOURCES ${R_SOURCES})
endif()

add_library(objxgboost OBJECT ${SOURCES})

# building shared library for R package
if(R_LIB)
  find_package(LibR REQUIRED)

  list(APPEND LINK_LIBRARIES "${LIBR_CORE_LIBRARY}")
  MESSAGE(STATUS "LIBR_CORE_LIBRARY " ${LIBR_CORE_LIBRARY})

  # Shared library target for the R package
  add_library(xgboost SHARED $<TARGET_OBJECTS:objxgboost>)
  include_directories(xgboost
    "${LIBR_INCLUDE_DIRS}"
    "${PROJECT_SOURCE_DIR}"
  )

  target_link_libraries(xgboost ${LINK_LIBRARIES})
  # R uses no lib prefix in shared library names of its packages
  set_target_properties(xgboost PROPERTIES PREFIX "")
  if(APPLE)
    set_target_properties(xgboost PROPERTIES SUFFIX ".so")
  endif()

  setup_rpackage_install_target(xgboost ${CMAKE_CURRENT_BINARY_DIR})
  # use a dummy location for any other remaining installs
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/dummy_inst")

  # main targets: shared library & exe
else()
  # Executable
  add_executable(runxgboost $<TARGET_OBJECTS:objxgboost> src/cli_main.cc)
  set_target_properties(runxgboost PROPERTIES
    OUTPUT_NAME xgboost
  )
  set_output_directory(runxgboost ${PROJECT_SOURCE_DIR})
  target_link_libraries(runxgboost ${LINK_LIBRARIES})

  # Shared library
  add_library(xgboost SHARED $<TARGET_OBJECTS:objxgboost>)
  target_link_libraries(xgboost ${LINK_LIBRARIES})
  set_output_directory(xgboost ${PROJECT_SOURCE_DIR}/lib)
  if(MINGW)
    # remove the 'lib' prefix to conform to windows convention for shared library names
    set_target_properties(xgboost PROPERTIES PREFIX "")
  endif()

  #Ensure these two targets do not build simultaneously, as they produce outputs with conflicting names
  add_dependencies(xgboost runxgboost)
endif()

# JVM
if(JVM_BINDINGS)
    find_package(JNI QUIET REQUIRED)

    add_library(xgboost4j SHARED
      $<TARGET_OBJECTS:objxgboost>
      jvm-packages/xgboost4j/src/native/xgboost4j.cpp)
    target_include_directories(xgboost4j
      PRIVATE ${JNI_INCLUDE_DIRS}
      PRIVATE jvm-packages/xgboost4j/src/native)
    target_link_libraries(xgboost4j
      ${LINK_LIBRARIES}
      ${JAVA_JVM_LIBRARY})
    set_output_directory(xgboost4j ${PROJECT_SOURCE_DIR}/lib)
endif()


# Test
if(GOOGLE_TEST)
  enable_testing()
  find_package(GTest REQUIRED)

  file(GLOB_RECURSE TEST_SOURCES "tests/cpp/*.cc")
  auto_source_group("${TEST_SOURCES}")

  if(USE_CUDA AND (NOT GENERATE_COMPILATION_DATABASE))
    file(GLOB_RECURSE CUDA_TEST_SOURCES "tests/cpp/*.cu")
    cuda_include_directories(${GTEST_INCLUDE_DIRS})
    cuda_compile(CUDA_TEST_OBJS ${CUDA_TEST_SOURCES})
  elseif (USE_CUDA AND GENERATE_COMPILATION_DATABASE)
    file(GLOB_RECURSE CUDA_TEST_SOURCES "tests/cpp/*.cu")
  else()
    set(CUDA_TEST_OBJS "")
  endif()

  if (USE_CUDA AND GENERATE_COMPILATION_DATABASE)
    add_executable(testxgboost ${TEST_SOURCES} ${CUDA_TEST_SOURCES}
      $<TARGET_OBJECTS:objxgboost>)
    target_include_directories(testxgboost PRIVATE cub)
  else ()
    add_executable(testxgboost ${TEST_SOURCES} ${CUDA_TEST_OBJS}
      $<TARGET_OBJECTS:objxgboost>)
  endif ()

  set_output_directory(testxgboost ${PROJECT_SOURCE_DIR})
  target_include_directories(testxgboost
    PRIVATE ${GTEST_INCLUDE_DIRS})
  target_link_libraries(testxgboost ${GTEST_LIBRARIES} ${LINK_LIBRARIES})

  add_test(TestXGBoost testxgboost)
endif()


# Group sources
auto_source_group("${SOURCES}")
