cmake_minimum_required(VERSION 2.8)

project(xgboost)
	
IF(APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-narrowing")
ENDIF(APPLE)

### ---[ Find universal dependencies
# the gcc-4.2.1 coming with MacOS X is not compatible with the OpenMP pragmas we use, so disabling OpenMP for it
if((NOT APPLE) OR (NOT CMAKE_COMPILER_IS_GNUCXX) OR (GCC_VERSION VERSION_GREATER 4.2.1) OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
find_package(OpenMP)
endif()
if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
message (STATUS "Found OpenMP")
if(MSVC90 OR MSVC10)
if(MSVC90)
set(OPENMP_DLL VCOMP90)
elseif(MSVC10)
set(OPENMP_DLL VCOMP100)
endif(MSVC90)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS_DEBUG} /DELAYLOAD:${OPENMP_DLL}D.dll")
set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /DELAYLOAD:${OPENMP_DLL}.dll")
endif(MSVC90 OR MSVC10)
else(OPENMP_FOUND)
message (STATUS "Not found OpenMP")
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support.")
endif()

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rabit)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rabit/include)
include_directories (${PROJECT_SOURCE_DIR}/includes) #todo use relative paths to project here!s
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/c_api)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/common)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/data)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/src/tree)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/dmlc-core/include)

add_subdirectory(dmlc-core)
add_subdirectory(rabit)

SET(xgboost_HDRS include/xgboost/data.h
	include/xgboost/c_api.h
	include/xgboost/base.h
	include/xgboost/feature_map.h
	include/xgboost/gbm.h
	include/xgboost/learner.h
	include/xgboost/logging.h
	include/xgboost/metric.h
	include/xgboost/objective.h
	include/xgboost/tree_model.h
	include/xgboost/tree_updater.h
	src/c_api/c_api_error.h
	src/common/base64.h
	src/common/config.h
	src/common/group_data.h
	src/common/io.h
	src/common/xgboost_math.h
	src/common/quantile.h
	src/common/random.h
	src/common/sync.h
	src/common/thread_local.h	
	src/common/common.h
	src/data/simple_csr_source.h
	src/data/simple_dmatrix.h
	src/data/simple_csr_source.h
	src/data/sparse_batch_page.h
	src/data/sparse_page_dmatrix.h
	src/data/sparse_page_source.h
	src/tree/param.h
	src/tree/updater_basemaker-inl.h
)  
SET(xgboost_SOURCES
	src/c_api/c_api_error.cc
	src/c_api/c_api.cc
	src/common/common.cc
	src/data/data.cc
	src/data/simple_csr_source.cc
	src/data/simple_dmatrix.cc
	src/data/sparse_page_dmatrix.cc
	src/data/sparse_page_raw_format.cc
	src/data/sparse_page_source.cc
	src/gbm/gblinear.cc
	src/gbm/gbtree.cc
	src/gbm/gbm.cc
	src/metric/elementwise_metric.cc
	src/metric/multiclass_metric.cc
	src/metric/metric.cc
	src/metric/rank_metric.cc
	src/objective/multiclass_obj.cc
	src/objective/objective.cc
	src/objective/rank_obj.cc
	src/objective/regression_obj.cc
	src/tree/tree_model.cc
	src/tree/tree_updater.cc
	src/tree/updater_colmaker.cc
	src/tree/updater_histmaker.cc
	src/tree/updater_prune.cc
	src/tree/updater_refresh.cc
	src/tree/updater_skmaker.cc
	src/tree/updater_sync.cc
	src/cli_main.cc
	src/learner.cc
	src/logging.cc
)

#include_directories(${xgboost_HDRS})
#subtree/rabit/src/engine_mpi.cc not needed

add_library(xgboost STATIC ${xgboost_SOURCES} ${xgboost_HDRS})
#target_link_libraries (rabit ${Boost_LIBRARIES} ${PCL_LIBRARIES})



