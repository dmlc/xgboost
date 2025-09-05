
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was xgboost-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(USE_OPENMP ON)
set(USE_CUDA OFF)
set(USE_NCCL OFF)
set(XGBOOST_BUILD_STATIC_LIB OFF)

include(CMakeFindDependencyMacro)

if (XGBOOST_BUILD_STATIC_LIB)
  find_dependency(Threads)
  if(USE_OPENMP)
    find_dependency(OpenMP)
  endif()
  if(USE_CUDA)
    find_dependency(CUDA)
  endif()
  # nccl should be linked statically if xgboost is built as static library.
endif (XGBOOST_BUILD_STATIC_LIB)

if(NOT TARGET xgboost::xgboost)
  include(${CMAKE_CURRENT_LIST_DIR}/XGBoostTargets.cmake)
endif()

message(STATUS "Found XGBoost (found version \"${xgboost_VERSION}\")")
