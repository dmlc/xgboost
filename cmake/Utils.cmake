# Automatically set source group based on folder
function(auto_source_group SOURCES)

  foreach(FILE ${SOURCES})
      get_filename_component(PARENT_DIR "${FILE}" PATH)

      # skip src or include and changes /'s to \\'s
      string(REPLACE "${CMAKE_CURRENT_LIST_DIR}" "" GROUP "${PARENT_DIR}")
      string(REPLACE "/" "\\\\" GROUP "${GROUP}")
      string(REGEX REPLACE "^\\\\" "" GROUP "${GROUP}")

      source_group("${GROUP}" FILES "${FILE}")
  endforeach()
endfunction(auto_source_group)

# Force static runtime for MSVC
function(msvc_use_static_runtime)
  if(MSVC AND (NOT BUILD_SHARED_LIBS) AND (NOT FORCE_SHARED_CRT))
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
              set(${variable} "${${variable}}"  PARENT_SCOPE)
          endif()
      endforeach()
      set(variables
          CMAKE_CUDA_FLAGS
          CMAKE_CUDA_FLAGS_DEBUG
          CMAKE_CUDA_FLAGS_MINSIZEREL
          CMAKE_CUDA_FLAGS_RELEASE
          CMAKE_CUDA_FLAGS_RELWITHDEBINFO
      )
      foreach(variable ${variables})
          if(${variable} MATCHES "-MD")
              string(REGEX REPLACE "-MD" "-MT" ${variable} "${${variable}}")
              set(${variable} "${${variable}}"  PARENT_SCOPE)
          endif()
          if(${variable} MATCHES "/MD")
              string(REGEX REPLACE "/MD" "/MT" ${variable} "${${variable}}")
              set(${variable} "${${variable}}"  PARENT_SCOPE)
          endif()
      endforeach()
  endif()
endfunction(msvc_use_static_runtime)

# Set output directory of target, ignoring debug or release
function(set_output_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${dir}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${dir}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${dir}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dir}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${dir}
    LIBRARY_OUTPUT_DIRECTORY ${dir}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${dir}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${dir}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dir}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${dir}
    ARCHIVE_OUTPUT_DIRECTORY ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${dir}
    ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${dir})
endfunction(set_output_directory)

# Set a default build type to release if none was specified
function(set_default_configuration_release)
    if(CMAKE_CONFIGURATION_TYPES STREQUAL "Debug;Release;MinSizeRel;RelWithDebInfo") # multiconfig generator?
        set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
	elseif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	  message(STATUS "Setting build type to 'Release' as none was specified.")
	  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE )
	endif()
endfunction(set_default_configuration_release)

# Generate nvcc compiler flags given a list of architectures
# Also generates PTX for the most recent architecture for forwards compatibility
function(format_gencode_flags flags out)
  if(CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
    set(CUDA_VERSION "${CMAKE_MATCH_1}")
  endif()
  # Set up architecture flags
  if(NOT flags)
    if (CUDA_VERSION VERSION_GREATER_EQUAL "11.1")
      set(flags "50;60;70;80")
    elseif (CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      set(flags "50;60;70;80")
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      set(flags "35;50;60;70")
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
      set(flags "35;50;60;70")
    else()
      set(flags "35;50;60")
    endif()
  endif()

  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    cmake_policy(SET CMP0104 NEW)
    list(GET flags -1 latest_arch)
    list(TRANSFORM flags APPEND "-real")
    list(APPEND flags ${latest_arch})
    set(CMAKE_CUDA_ARCHITECTURES ${flags})
    set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
  else()
    # Generate SASS
    foreach(ver ${flags})
      set(${out} "${${out}}--generate-code=arch=compute_${ver},code=sm_${ver};")
    endforeach()
    # Generate PTX for last architecture
    list(GET flags -1 ver)
    set(${out} "${${out}}--generate-code=arch=compute_${ver},code=compute_${ver};")
    set(${out} "${${out}}" PARENT_SCOPE)
    message(STATUS "CUDA GEN_CODE: ${GEN_CODE}")
  endif (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
endfunction(format_gencode_flags flags)

macro(enable_nvtx target)
  find_package(NVTX REQUIRED)
  target_include_directories(${target} PRIVATE "${NVTX_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${NVTX_LIBRARY}")
  target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_NVTX=1)
endmacro()

# Set CUDA related flags to target.  Must be used after code `format_gencode_flags`.
function(xgboost_set_cuda_flags target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:${GEN_CODE}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>)

  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  endif (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")

  if (FORCE_COLORED_OUTPUT)
    if (FORCE_COLORED_OUTPUT AND (CMAKE_GENERATOR STREQUAL "Ninja") AND
        ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
          (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fdiagnostics-color=always>)
    endif()
  endif (FORCE_COLORED_OUTPUT)

  if (USE_DEVICE_DEBUG)
    target_compile_options(${target} PRIVATE
      $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>)
  else (USE_DEVICE_DEBUG)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif (USE_DEVICE_DEBUG)

  if (USE_NVTX)
    enable_nvtx(${target})
  endif (USE_NVTX)

  target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_CUDA=1)
  target_include_directories(${target} PRIVATE ${xgboost_SOURCE_DIR}/gputreeshap)

  if (MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  endif (MSVC)

  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF)
endfunction(xgboost_set_cuda_flags)

macro(xgboost_link_nccl target)
  if (BUILD_STATIC_LIB)
    target_include_directories(${target} PUBLIC ${NCCL_INCLUDE_DIR})
    target_compile_definitions(${target} PUBLIC -DXGBOOST_USE_NCCL=1)
    target_link_libraries(${target} PUBLIC ${NCCL_LIBRARY})
  else ()
    target_include_directories(${target} PRIVATE ${NCCL_INCLUDE_DIR})
    target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_NCCL=1)
    target_link_libraries(${target} PRIVATE ${NCCL_LIBRARY})
  endif (BUILD_STATIC_LIB)
endmacro(xgboost_link_nccl)

# compile options
macro(xgboost_target_properties target)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

  if (HIDE_CXX_SYMBOLS)
    #-- Hide all C++ symbols
    set_target_properties(${target} PROPERTIES
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
      CUDA_VISIBILITY_PRESET hidden
    )
  endif (HIDE_CXX_SYMBOLS)

  if (ENABLE_ALL_WARNINGS)
    target_compile_options(${target} PUBLIC
      $<IF:$<COMPILE_LANGUAGE:CUDA>,
      -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wno-expansion-to-defined,
      -Wall -Wextra -Wno-expansion-to-defined>
    )
  endif(ENABLE_ALL_WARNINGS)

  target_compile_options(${target}
    PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/MP>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:-funroll-loops>)

  if (MSVC)
    target_compile_options(${target} PRIVATE
      $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>
      -D_CRT_SECURE_NO_WARNINGS
      -D_CRT_SECURE_NO_DEPRECATE
    )
  endif (MSVC)

  if (WIN32 AND MINGW)
    target_compile_options(${target} PUBLIC -static-libstdc++)
  endif (WIN32 AND MINGW)
endmacro(xgboost_target_properties)

# Custom definitions used in xgboost.
macro(xgboost_target_defs target)
  if (NOT ${target} STREQUAL "dmlc") # skip dmlc core for custom logging.
    target_compile_definitions(${target}
      PRIVATE
      -DDMLC_LOG_CUSTOMIZE=1
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:_MWAITXINTRIN_H_INCLUDED>)
  endif ()
  if (USE_DEBUG_OUTPUT)
    target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_DEBUG_OUTPUT=1)
  endif (USE_DEBUG_OUTPUT)
  if (XGBOOST_MM_PREFETCH_PRESENT)
    target_compile_definitions(${target}
      PRIVATE
      -DXGBOOST_MM_PREFETCH_PRESENT=1)
  endif(XGBOOST_MM_PREFETCH_PRESENT)
  if (XGBOOST_BUILTIN_PREFETCH_PRESENT)
    target_compile_definitions(${target}
      PRIVATE
      -DXGBOOST_BUILTIN_PREFETCH_PRESENT=1)
  endif (XGBOOST_BUILTIN_PREFETCH_PRESENT)

  if (PLUGIN_RMM)
    target_compile_definitions(objxgboost PUBLIC -DXGBOOST_USE_RMM=1)
  endif (PLUGIN_RMM)
endmacro(xgboost_target_defs)

# handles dependencies
macro(xgboost_target_link_libraries target)
  if (BUILD_STATIC_LIB)
    target_link_libraries(${target} PUBLIC Threads::Threads ${CMAKE_THREAD_LIBS_INIT})
  else()
    target_link_libraries(${target} PRIVATE Threads::Threads ${CMAKE_THREAD_LIBS_INIT})
  endif (BUILD_STATIC_LIB)

  if (USE_OPENMP)
    if (BUILD_STATIC_LIB)
      target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    else()
      target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif (BUILD_STATIC_LIB)
  endif (USE_OPENMP)

  if (USE_CUDA)
    xgboost_set_cuda_flags(${target})
  endif (USE_CUDA)

  if (PLUGIN_RMM)
    target_link_libraries(${target} PRIVATE rmm::rmm)
  endif (PLUGIN_RMM)

  if (USE_NCCL)
    xgboost_link_nccl(${target})
  endif (USE_NCCL)

  if (USE_NVTX)
    enable_nvtx(${target})
  endif (USE_NVTX)

  if (RABIT_BUILD_MPI)
    target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
  endif (RABIT_BUILD_MPI)

  if (MINGW)
    target_link_libraries(${target} PRIVATE wsock32 ws2_32)
  endif (MINGW)
endmacro(xgboost_target_link_libraries)
