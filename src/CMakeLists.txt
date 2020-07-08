file(GLOB_RECURSE CPU_SOURCES *.cc *.h)
list(REMOVE_ITEM CPU_SOURCES ${xgboost_SOURCE_DIR}/src/cli_main.cc)

#-- Object library
# Object library is necessary for jvm-package, which creates its own shared library.
add_library(objxgboost OBJECT)
target_sources(objxgboost PRIVATE ${CPU_SOURCES})
if (USE_CUDA)
  file(GLOB_RECURSE CUDA_SOURCES *.cu *.cuh)
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    target_include_directories(objxgboost PRIVATE ${xgboost_SOURCE_DIR}/cub/)
  endif (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
  target_compile_options(objxgboost PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:${GEN_CODE}>)

  if (USE_NCCL)
    find_package(Nccl REQUIRED)
    target_include_directories(objxgboost PRIVATE ${NCCL_INCLUDE_DIR})
    target_compile_definitions(objxgboost PRIVATE -DXGBOOST_USE_NCCL=1)
    target_link_libraries(objxgboost PUBLIC ${NCCL_LIBRARY})
  endif (USE_NCCL)

  if (USE_NVTX)
    enable_nvtx(objxgboost)
  endif (USE_NVTX)

  target_compile_options(objxgboost PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
  )

  if (MSVC)
    target_compile_options(objxgboost PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
  endif (MSVC)

  if (HIDE_CXX_SYMBOLS)
    target_compile_options(objxgboost PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>
    )
  endif (HIDE_CXX_SYMBOLS)

  set_target_properties(objxgboost PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF)
endif (USE_CUDA)

target_include_directories(objxgboost
  PRIVATE
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include
  ${xgboost_SOURCE_DIR}/rabit/include)
target_compile_options(objxgboost
  PRIVATE
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/MP>
  $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:-funroll-loops>)
if (WIN32 AND MINGW)
  target_compile_options(objxgboost PUBLIC -static-libstdc++)
endif (WIN32 AND MINGW)

if (MSVC)
  target_compile_options(objxgboost PRIVATE
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_SECURE_NO_DEPRECATE
  )
endif (MSVC)

set_target_properties(objxgboost PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)
target_compile_definitions(objxgboost
  PRIVATE
  -DDMLC_LOG_CUSTOMIZE=1  # enable custom logging
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:_MWAITXINTRIN_H_INCLUDED>)
if (USE_DEBUG_OUTPUT)
  target_compile_definitions(objxgboost PRIVATE -DXGBOOST_USE_DEBUG_OUTPUT=1)
endif (USE_DEBUG_OUTPUT)

if (XGBOOST_MM_PREFETCH_PRESENT)
  target_compile_definitions(objxgboost
    PRIVATE
    -DXGBOOST_MM_PREFETCH_PRESENT=1)
endif(XGBOOST_MM_PREFETCH_PRESENT)
if (XGBOOST_BUILTIN_PREFETCH_PRESENT)
  target_compile_definitions(objxgboost
    PRIVATE
    -DXGBOOST_BUILTIN_PREFETCH_PRESENT=1)
endif (XGBOOST_BUILTIN_PREFETCH_PRESENT)

find_package(Threads REQUIRED)
target_link_libraries(objxgboost PUBLIC Threads::Threads ${CMAKE_THREAD_LIBS_INIT})

if (USE_OPENMP OR USE_CUDA)  # CUDA requires OpenMP
  find_package(OpenMP REQUIRED)
  target_link_libraries(objxgboost PUBLIC OpenMP::OpenMP_CXX)
endif (USE_OPENMP OR USE_CUDA)

# For MSVC: Call msvc_use_static_runtime() once again to completely
# replace /MD with /MT. See https://github.com/dmlc/xgboost/issues/4462
# for issues caused by mixing of /MD and /MT flags
msvc_use_static_runtime()

# This grouping organises source files nicely in visual studio
auto_source_group("${CUDA_SOURCES}")
auto_source_group("${CPU_SOURCES}")

#-- End object library
