find_package(GTest REQUIRED)
file(GLOB_RECURSE TEST_SOURCES "*.cc")

if (USE_CUDA)
  find_package(Nccl REQUIRED)
  file(GLOB_RECURSE CUDA_TEST_SOURCES "*.cu")
  list(APPEND TEST_SOURCES ${CUDA_TEST_SOURCES})
endif (USE_CUDA)
add_executable(testxgboost ${TEST_SOURCES} ${XGBOOST_OBJ_SOURCES})

if (USE_CUDA)
  target_include_directories(testxgboost PRIVATE
    ${PROJECT_SOURCE_DIR}/cub/)
  target_compile_options(testxgboost PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<COMPILE_LANGUAGE:CUDA>:--std=c++11>
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
    target_include_directories(testxgboost PRIVATE "${NVTX_HEADER_DIR}")
    target_compile_definitions(testxgboost PRIVATE -DXGBOOST_USE_NVTX=1)
  endif (USE_NVTX)
endif (USE_CUDA)

target_include_directories(testxgboost
  PRIVATE
  ${GTEST_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/dmlc-core/include
  ${PROJECT_SOURCE_DIR}/rabit/include)
set_target_properties(
  testxgboost PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON)
target_link_libraries(testxgboost
  PRIVATE
  ${GTEST_LIBRARIES}
  ${LINKED_LIBRARIES_PRIVATE}
  ${OpenMP_CXX_LIBRARIES})
target_compile_definitions(testxgboost PRIVATE ${XGBOOST_DEFINITIONS})
if (USE_OPENMP)
  target_compile_options(testxgboost PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
endif (USE_OPENMP)
set_output_directory(testxgboost ${PROJECT_BINARY_DIR})
