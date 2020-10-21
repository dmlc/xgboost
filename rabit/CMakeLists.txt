cmake_minimum_required(VERSION 3.3)

find_package(Threads REQUIRED)

set(RABIT_SOURCES
  ${CMAKE_CURRENT_LIST_DIR}/src/allreduce_base.cc
  ${CMAKE_CURRENT_LIST_DIR}/src/c_api.cc)

if (RABIT_BUILD_MPI)
  list(APPEND RABIT_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/engine_mpi.cc)
elseif (RABIT_MOCK)
  list(APPEND RABIT_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/engine_mock.cc)
else ()
  list(APPEND RABIT_SOURCES ${CMAKE_CURRENT_LIST_DIR}/src/engine.cc)
endif ()

add_library(rabit OBJECT ${RABIT_SOURCES})
target_link_libraries(rabit Threads::Threads dmlc)

if (MSVC)
  target_compile_options(rabit PRIVATE
    -D_CRT_SECURE_NO_WARNINGS -D_CRT_SECURE_NO_DEPRECATE)
endif (MSVC)
if (ENABLE_ALL_WARNINGS)
  target_compile_options(rabit PRIVATE -Wall -Wextra)
endif (ENABLE_ALL_WARNINGS)

set_target_properties(rabit
  PROPERTIES CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON)

if(RABIT_BUILD_MPI)
  find_package(MPI REQUIRED)
  if (NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "CXX Interface for MPI is required for building MPI backend.")
  endif (NOT MPI_CXX_FOUND)
  target_link_libraries(rabit ${MPI_CXX_LIBRARIES})
endif()

target_include_directories(rabit PUBLIC
  "$<BUILD_INTERFACE:${xgboost_SOURCE_DIR}/rabit/include>"
  "$<BUILD_INTERFACE:${xgboost_SOURCE_DIR}/dmlc-core/include>")
target_include_directories(rabit PRIVATE
  ${xgboost_SOURCE_DIR}/include)
