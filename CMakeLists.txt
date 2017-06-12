cmake_minimum_required (VERSION 3.2)
project (xgboost)
find_package(OpenMP)

option(PLUGIN_UPDATER_GPU "Build GPU accelerated tree construction plugin")
set(GPU_COMPUTE_VER 50;52;60;61 CACHE STRING
    "Space separated list of compute versions to be built against")
if(PLUGIN_UPDATER_GPU)
  find_package(CUDA REQUIRED)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -O3 -funroll-loops -msse2 -D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES")
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

  set(STD_FLAG "")
  if(COMPILER_SUPPORTS_CXX11)
    set(STD_FLAG "-std=c++11")
  elseif(COMPILER_SUPPORTS_CXX0X)
    set(STD_FLAG "-std=c++0x")
  else()
    message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  endif()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${STD_FLAG}")
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

set(LINK_LIBRARIES dmlccore rabit)

if(PLUGIN_UPDATER_GPU)
    # nccl
    set(LINK_LIBRARIES ${LINK_LIBRARIES} nccl)
    add_subdirectory(nccl)
    set(NCCL_DIRECTORY ${PROJECT_SOURCE_DIR}/nccl)
    include_directories(${NCCL_DIRECTORY}/src)
    set(LINK_LIBRARIES ${LINK_LIBRARIES} ${CUDA_LIBRARIES})
    #Find cub
    set(CUB_DIRECTORY ${PROJECT_SOURCE_DIR}/cub/)
    include_directories(${CUB_DIRECTORY})
    #Find googletest
    set(GTEST_DIRECTORY "${CACHE_PREFIX}" CACHE PATH "Googletest directory")
    include_directories(${GTEST_DIRECTORY}/include)
    #gencode flags
    set(GENCODE_FLAGS "")
    foreach(ver ${GPU_COMPUTE_VER})
      set(GENCODE_FLAGS "${GENCODE_FLAGS}-gencode arch=compute_${ver},code=sm_${ver};")
    endforeach()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};--expt-extended-lambda;${GENCODE_FLAGS};-lineinfo;")
    if(NOT MSVC)
      set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler -fPIC")
    endif()
    set(CUDA_SOURCES
      plugin/updater_gpu/src/updater_gpu.cu
      plugin/updater_gpu/src/gpu_hist_builder.cu
      )
    # use below for forcing specific arch
    cuda_compile(CUDA_OBJS ${CUDA_SOURCES} ${CUDA_NVCC_FLAGS})
else()
    set(CUDA_OBJS "")
endif()

add_library(objxgboost OBJECT ${SOURCES})
set_target_properties(${objxgboost} PROPERTIES POSITION_INDEPENDENT_CODE 1)

add_library(libxgboost SHARED $<TARGET_OBJECTS:objxgboost> ${CUDA_OBJS})
add_executable(xgboost $<TARGET_OBJECTS:objxgboost> ${CUDA_OBJS})

target_link_libraries(xgboost ${LINK_LIBRARIES})
target_link_libraries(libxgboost ${LINK_LIBRARIES})

option(JVM_BINDINGS "Build JVM bindings" OFF)

if(JVM_BINDINGS)
    find_package(JNI QUIET REQUIRED)

    include_directories(${JNI_INCLUDE_DIRS} jvm-packages/xgboost4j/src/native)

    add_library(libxgboost4j SHARED
        $<TARGET_OBJECTS:objxgboost>
        ${CUDA_OBJS}
        jvm-packages/xgboost4j/src/native/xgboost4j.cpp)
    target_link_libraries(libxgboost4j
        ${LINK_LIBRARIES}
        ${JNI_LIBRARIES})
endif()
