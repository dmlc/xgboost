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
endfunction()

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
endfunction()

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
endfunction()

# Set a default build type to release if none was specified
function(set_default_configuration_release)
    if(CMAKE_CONFIGURATION_TYPES STREQUAL "Debug;Release;MinSizeRel;RelWithDebInfo") # multiconfig generator?
        set(CMAKE_CONFIGURATION_TYPES Release CACHE STRING "" FORCE)
    elseif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
      message(STATUS "Setting build type to 'Release' as none was specified.")
      set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    endif()
endfunction()

# Generate CMAKE_CUDA_ARCHITECTURES form a list of architectures
# Also generates PTX for the most recent architecture for forwards compatibility
function(compute_cmake_cuda_archs archs)
  if(CMAKE_CUDA_COMPILER_VERSION MATCHES "^([0-9]+\\.[0-9]+)")
    set(CUDA_VERSION "${CMAKE_MATCH_1}")
  endif()
  list(SORT archs)
  unset(CMAKE_CUDA_ARCHITECTURES CACHE)
  set(CMAKE_CUDA_ARCHITECTURES ${archs})

  # Set up defaults based on CUDA varsion
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CUDA_VERSION VERSION_GREATER_EQUAL "11.8")
      set(CMAKE_CUDA_ARCHITECTURES 50 60 70 80 90)
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      set(CMAKE_CUDA_ARCHITECTURES 50 60 70 80)
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      set(CMAKE_CUDA_ARCHITECTURES 35 50 60 70)
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
      set(CMAKE_CUDA_ARCHITECTURES 35 50 60 70)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 35 50 60)
    endif()
  endif()

  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES APPEND "-real")
  list(TRANSFORM CMAKE_CUDA_ARCHITECTURES REPLACE "([0-9]+)-real" "\\0;\\1-virtual" AT -1)
  set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" PARENT_SCOPE)
  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endfunction()

# Set CUDA related flags to target.  Must be used after code `format_gencode_flags`.
function(xgboost_set_cuda_flags target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xfatbin=-compress-all>)

  if(USE_PER_THREAD_DEFAULT_STREAM)
    target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>)
  endif()

  if(FORCE_COLORED_OUTPUT)
    if(FORCE_COLORED_OUTPUT AND (CMAKE_GENERATOR STREQUAL "Ninja") AND
        ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") OR
          (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")))
      target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fdiagnostics-color=always>)
    endif()
  endif()

  if(USE_DEVICE_DEBUG)
    target_compile_options(${target} PRIVATE
      $<$<AND:$<CONFIG:DEBUG>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>)
  else()
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  endif()

  if(USE_NVTX)
    target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_NVTX=1)
  endif()

  target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_CUDA=1)
  target_include_directories(
    ${target} PRIVATE
    ${xgboost_SOURCE_DIR}/gputreeshap
    ${CUDAToolkit_INCLUDE_DIRS})

  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  endif()

  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  if(USE_CUDA_LTO)
    set_target_properties(${target} PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION ON
      CUDA_SEPARABLE_COMPILATION ON)
  else()
    set_target_properties(${target} PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF)
  endif()
endfunction()

function(xgboost_link_nccl target)
  set(xgboost_nccl_flags -DXGBOOST_USE_NCCL=1)
  if(USE_DLOPEN_NCCL)
    list(APPEND xgboost_nccl_flags -DXGBOOST_USE_DLOPEN_NCCL=1)
  endif()

  if(BUILD_STATIC_LIB)
    target_include_directories(${target} PUBLIC ${NCCL_INCLUDE_DIR})
    target_compile_definitions(${target} PUBLIC ${xgboost_nccl_flags})
    target_link_libraries(${target} PUBLIC ${NCCL_LIBRARY})
  else()
    target_include_directories(${target} PRIVATE ${NCCL_INCLUDE_DIR})
    target_compile_definitions(${target} PRIVATE ${xgboost_nccl_flags})
    if(NOT USE_DLOPEN_NCCL)
      target_link_libraries(${target} PRIVATE ${NCCL_LIBRARY})
    endif()
  endif()
endfunction()

# compile options
macro(xgboost_target_properties target)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)

  if(HIDE_CXX_SYMBOLS)
    #-- Hide all C++ symbols
    set_target_properties(${target} PROPERTIES
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
      CUDA_VISIBILITY_PRESET hidden
    )
  endif()

  if(ENABLE_ALL_WARNINGS)
    target_compile_options(${target} PUBLIC
      $<IF:$<COMPILE_LANGUAGE:CUDA>,
      -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wno-expansion-to-defined,
      -Wall -Wextra -Wno-expansion-to-defined>
    )
  endif()

  target_compile_options(${target}
    PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/MP>
    $<$<AND:$<NOT:$<CXX_COMPILER_ID:MSVC>>,$<COMPILE_LANGUAGE:CXX>>:-funroll-loops>)

  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>
      -D_CRT_SECURE_NO_WARNINGS
      -D_CRT_SECURE_NO_DEPRECATE
    )
  endif()

  if(WIN32 AND MINGW)
    target_compile_options(${target} PUBLIC -static-libstdc++)
  endif()
endmacro()

# Custom definitions used in xgboost.
macro(xgboost_target_defs target)
  if(NOT ${target} STREQUAL "dmlc") # skip dmlc core for custom logging.
    target_compile_definitions(${target}
      PRIVATE
      -DDMLC_LOG_CUSTOMIZE=1
      $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:_MWAITXINTRIN_H_INCLUDED>)
  endif()
  if(USE_DEBUG_OUTPUT)
    target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_DEBUG_OUTPUT=1)
  endif()
  if(XGBOOST_MM_PREFETCH_PRESENT)
    target_compile_definitions(${target}
      PRIVATE
      -DXGBOOST_MM_PREFETCH_PRESENT=1)
  endif()
  if(XGBOOST_BUILTIN_PREFETCH_PRESENT)
    target_compile_definitions(${target}
      PRIVATE
      -DXGBOOST_BUILTIN_PREFETCH_PRESENT=1)
  endif()

  if(PLUGIN_RMM)
    target_compile_definitions(objxgboost PUBLIC -DXGBOOST_USE_RMM=1)
  endif()
endmacro()

# handles dependencies
macro(xgboost_target_link_libraries target)
  if(BUILD_STATIC_LIB)
    target_link_libraries(${target} PUBLIC Threads::Threads ${CMAKE_THREAD_LIBS_INIT})
  else()
    target_link_libraries(${target} PRIVATE Threads::Threads ${CMAKE_THREAD_LIBS_INIT})
  endif()

  if(USE_OPENMP)
    if(BUILD_STATIC_LIB)
      target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)
    else()
      target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
  endif()

  if(USE_CUDA)
    xgboost_set_cuda_flags(${target})
    target_link_libraries(${target} PUBLIC CUDA::cudart_static)
  endif()

  if(PLUGIN_RMM)
    target_link_libraries(${target} PRIVATE rmm::rmm)
  endif()

  if(USE_NCCL)
    xgboost_link_nccl(${target})
  endif()

  if(USE_NVTX)
    target_link_libraries(${target} PRIVATE CUDA::nvToolsExt)
  endif()

  if(MINGW)
    target_link_libraries(${target} PRIVATE wsock32 ws2_32)
  endif()
endmacro()
