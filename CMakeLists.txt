cmake_minimum_required (VERSION 2.6)
project (xgboost)
find_package(OpenMP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()


# Make sure we are using C++11
# Visual Studio 12.0 and newer supports enough c++11 to make this work
if(MSVC)
  if(MSVC_VERSION LESS 1800)
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  endif()
else()
  # GCC 4.6 with c++0x supports enough to make this work
  include(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)

  if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  endif()
endif()

#Make sure we are using the static runtime
if(MSVC)
    set(variables
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_MINSIZEREL
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_MINSIZEREL
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
    )
    foreach(variable ${variables})
        if(${variable} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
        endif()
    endforeach()
endif()

include_directories (
    ${PROJECT_SOURCE_DIR}/include 
    ${PROJECT_SOURCE_DIR}/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/rabit/include 
    )

file(GLOB SOURCES 
    src/c_api/*.cc
    src/common/*.cc
    src/data/*.cc
    src/gbm/*.cc
    src/metric/*.cc
    src/objective/*.cc
    src/tree/*.cc
    src/*.cc
)

set(RABIT_SOURCES
    rabit/src/allreduce_base.cc
    rabit/src/allreduce_robust.cc
    rabit/src/engine.cc
    rabit/src/c_api.cc
)

add_subdirectory(dmlc-core) 

add_library(rabit STATIC ${RABIT_SOURCES})

#Set library output directories
if(MSVC)
    #With MSVC shared library is considered runtime
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
    #Prevent shared library being called liblibxgboost.so on Linux
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif()

option(PLUGIN_UPDATER_GPU "Build GPU accelerated tree construction plugin")

if(PLUGIN_UPDATER_GPU)
	#Find cub
	set(CUB_DIRECTORY "" CACHE PATH "CUB 1.5.4 directory")
	include_directories(${CUB_DIRECTORY})
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;-arch=compute_35")
    if(NOT MSVC)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC")
    endif()
    set(SOURCES ${SOURCES} 
        plugin/updater_gpu/src/updater_gpu.cc
    )
    find_package(CUDA QUIET REQUIRED)
    cuda_add_library(updater_gpu STATIC 
        plugin/updater_gpu/src/gpu_builder.cu
    )
endif()

add_executable(xgboost ${SOURCES})
add_library(libxgboost SHARED ${SOURCES})
target_link_libraries(xgboost dmlccore rabit)
target_link_libraries(libxgboost dmlccore rabit)


if(PLUGIN_UPDATER_GPU)
    target_link_libraries(xgboost updater_gpu)
    target_link_libraries(libxgboost updater_gpu)
endif()

