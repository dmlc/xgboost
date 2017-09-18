cmake_minimum_required (VERSION 3.2)
project(xgboost)
include(cmake/Utils.cmake)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
find_package(OpenMP)

set_default_configuration_release()
msvc_use_static_runtime()

# Options
option(USE_CUDA  "Build with GPU acceleration") 
option(JVM_BINDINGS "Build JVM bindings" OFF)
option(GOOGLE_TEST "Build google tests" OFF)
option(R_LIB "Build shared library for R package" OFF)
set(GPU_COMPUTE_VER 35;50;52;60;61 CACHE STRING
  "Space separated list of compute versions to be built against")

# Deprecation warning
if(PLUGIN_UPDATER_GPU)
  set(USE_CUDA ON)
  message(WARNING "The option 'PLUGIN_UPDATER_GPU' is deprecated. Set 'USE_CUDA' instead.")
endif()

# Compiler flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
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

include_directories (
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/rabit/include
)

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

# rabit
# TODO: Create rabit cmakelists.txt
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
if(MINGW)
  # build a dummy rabit library
  add_library(rabit STATIC ${RABIT_EMPTY_SOURCES})
else()
  add_library(rabit STATIC ${RABIT_SOURCES})
endif()


# dmlc-core
add_subdirectory(dmlc-core)
set(LINK_LIBRARIES dmlccore rabit)

if(USE_CUDA)
  find_package(CUDA 7.5 REQUIRED)
  cmake_minimum_required(VERSION 3.5)

  add_definitions(-DXGBOOST_USE_CUDA)
  
  include_directories(
    nccl/src
    cub
  )

  set(GENCODE_FLAGS "")
  format_gencode_flags("${GPU_COMPUTE_VER}" GENCODE_FLAGS)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;${GENCODE_FLAGS};-lineinfo;")
  if(NOT MSVC)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC; -std=c++11")
  endif()

  add_subdirectory(nccl)
  cuda_add_library(gpuxgboost ${CUDA_SOURCES} STATIC)
  target_link_libraries(gpuxgboost nccl)
  list(APPEND LINK_LIBRARIES gpuxgboost) 
endif()


# building shared library for R package
if(R_LIB)

  if(WIN32 AND MSVC)
    # allow .dll as well
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".lib" ".dll")
  endif()
  #include(cmake/modules/FindLibR.cmake)
  find_package(LibR REQUIRED)

  # compiled code customizations for R package
  add_definitions(
    -DXGBOOST_STRICT_R_MODE=1
    -DXGBOOST_CUSTOMIZE_GLOBAL_PRNG=1
    -DDMLC_LOG_BEFORE_THROW=0
    -DDMLC_DISABLE_STDIN=1
    -DDMLC_LOG_CUSTOMIZE=1
    -DRABIT_CUSTOMIZE_MSG_
    -DRABIT_STRICT_CXX98_
  )
  file(GLOB_RECURSE R_SOURCES
    R-package/src/*.h
    R-package/src/*.c
    R-package/src/*.cc
  )
  list(APPEND SOURCES ${R_SOURCES})
  include_directories(
    "${LIBR_INCLUDE_DIRS}"
    "${PROJECT_SOURCE_DIR}"
  )

  if(WIN32 AND MSVC)
    # create a local R.lib import library for R.dll if it doesn't exist
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/R.lib")
      create_rlib_for_msvc()
    endif()
    set(LIBR_CORE_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/R.lib")
  endif()
  list(APPEND LINK_LIBRARIES "${LIBR_CORE_LIBRARY}")
  #MESSAGE(STATUS "LIBR_EXECUTABLE " ${LIBR_EXECUTABLE})
  MESSAGE(STATUS "LIBR_CORE_LIBRARY " ${LIBR_CORE_LIBRARY})

  add_library(objxgboost OBJECT ${SOURCES})

  # Shared library for the use with R
  add_library(xgboost SHARED $<TARGET_OBJECTS:objxgboost>)
  target_link_libraries(xgboost ${LINK_LIBRARIES})
  if(MINGW)
    # remove the 'lib' prefix to conform to windows convention for shared library names
    set_target_properties(xgboost PROPERTIES PREFIX "")
  endif()

  # Install target for the R package
  # assemble the R package files in the build area, and run R CMD INSTALL
  install(CODE "file(REMOVE_RECURSE \"${CMAKE_CURRENT_BINARY_DIR}/R-package\")")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/R-package"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
    REGEX "src/*" EXCLUDE
    REGEX "R-package/configure" EXCLUDE
  )
  install(TARGETS xgboost LIBRARY RUNTIME DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/R-package/src/")
  install(CODE "file(WRITE \"${CMAKE_CURRENT_BINARY_DIR}/R-package/src/Makevars\" \"all:\")")
  install(CODE "file(WRITE \"${CMAKE_CURRENT_BINARY_DIR}/R-package/src/Makevars.win\" \"all:\")")
  #install(SCRIPT "${PROJECT_SOURCE_DIR}/cmake/RpackPreInstall.cmake")
  install(CODE "execute_process(COMMAND \"${LIBR_EXECUTABLE}\" CMD INSTALL\
    \"--no-multiarch\" \"${CMAKE_CURRENT_BINARY_DIR}/R-package\")")

# main targets: shared library & exe
else()
  add_library(objxgboost OBJECT ${SOURCES})

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

    include_directories(${JNI_INCLUDE_DIRS} jvm-packages/xgboost4j/src/native)

    add_library(xgboost4j SHARED
        $<TARGET_OBJECTS:objxgboost>
        jvm-packages/xgboost4j/src/native/xgboost4j.cpp)
    set_output_directory(xgboost4j ${PROJECT_SOURCE_DIR}/lib)
    target_link_libraries(xgboost4j
        ${LINK_LIBRARIES}
        ${JAVA_JVM_LIBRARY})
endif()

# Test
if(GOOGLE_TEST)
  enable_testing()
  find_package(GTest REQUIRED)

  file(GLOB_RECURSE TEST_SOURCES "tests/cpp/*.cc")
  auto_source_group("${TEST_SOURCES}")
  include_directories(${GTEST_INCLUDE_DIRS})

  if(USE_CUDA)
    file(GLOB_RECURSE CUDA_TEST_SOURCES "tests/cpp/*.cu")
    cuda_compile(CUDA_TEST_OBJS ${CUDA_TEST_SOURCES})
  else()
    set(CUDA_TEST_OBJS "")
  endif()

  add_executable(testxgboost ${TEST_SOURCES} ${CUDA_TEST_OBJS} $<TARGET_OBJECTS:objxgboost>)
  set_output_directory(testxgboost ${PROJECT_SOURCE_DIR})
  target_link_libraries(testxgboost ${GTEST_BOTH_LIBRARIES} ${LINK_LIBRARIES})

  add_test(TestXGBoost testxgboost)
endif()

# Group sources
auto_source_group("${SOURCES}")
