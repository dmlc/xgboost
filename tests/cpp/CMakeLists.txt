if (USE_DMLC_GTEST)
  if (NOT TARGET gtest)
    message(FATAL_ERROR "USE_DMLC_GTEST=ON but dmlc-core didn't bundle gtest")
  endif (NOT TARGET gtest)
  set(GTEST_LIBRARIES gtest)
else (USE_DMLC_GTEST)
  find_package(GTest REQUIRED)
endif (USE_DMLC_GTEST)
file(GLOB_RECURSE TEST_SOURCES "*.cc")

if (USE_CUDA)
  file(GLOB_RECURSE CUDA_TEST_SOURCES "*.cu")
  list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCES})
endif (USE_CUDA)
add_executable(testxgboost ${TEST_SOURCES} ${XGBOOST_OBJ_SOURCES}
  ${xgboost_SOURCE_DIR}/plugin/example/custom_obj.cc)

if (USE_CUDA)
  # OpenMP is mandatory for CUDA
  find_package(OpenMP REQUIRED)
  target_include_directories(testxgboost PRIVATE
    ${xgboost_SOURCE_DIR}/cub/)
  target_compile_options(testxgboost PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${GEN_CODE}>)
  target_compile_definitions(testxgboost
    PRIVATE -DXGBOOST_USE_CUDA=1)
  set_target_properties(testxgboost PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF)

  if (USE_NCCL)
    find_package(Nccl REQUIRED)
    target_include_directories(testxgboost PRIVATE ${NCCL_INCLUDE_DIR})
    target_compile_definitions(testxgboost PRIVATE -DXGBOOST_USE_NCCL=1)
    target_link_libraries(testxgboost PRIVATE ${NCCL_LIBRARY})
  endif (USE_NCCL)

  if (USE_NVTX)
    enable_nvtx(testxgboost)
  endif (USE_NVTX)

  if (MSVC)
    target_compile_options(testxgboost PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>
    )
  endif (MSVC)

  set_target_properties(testxgboost PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON)
endif (USE_CUDA)

if (MSVC)
  target_compile_options(testxgboost PRIVATE
    $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>
    -D_CRT_SECURE_NO_WARNINGS
    -D_CRT_SECURE_NO_DEPRECATE
  )
endif (MSVC)

target_include_directories(testxgboost
  PRIVATE
  ${GTEST_INCLUDE_DIRS}
  ${xgboost_SOURCE_DIR}/include
  ${xgboost_SOURCE_DIR}/dmlc-core/include
  ${xgboost_SOURCE_DIR}/rabit/include)
set_target_properties(
  testxgboost PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(testxgboost
  PRIVATE
  ${GTEST_LIBRARIES}
  ${LINKED_LIBRARIES_PRIVATE})

target_compile_definitions(testxgboost PRIVATE ${XGBOOST_DEFINITIONS})
set_output_directory(testxgboost ${xgboost_BINARY_DIR})

# This grouping organises source files nicely in visual studio
auto_source_group("${TEST_SOURCES}")
