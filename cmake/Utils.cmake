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
    if (CUDA_VERSION VERSION_GREATER_EQUAL "11.0")
      set(flags "35;50;52;60;61;70;75;80")
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      set(flags "35;50;52;60;61;70;75")
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
      set(flags "35;50;52;60;61;70")
    else()
      set(flags "35;50;52;60;61")
    endif()
  endif()

  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    cmake_policy(SET CMP0104 NEW)
    foreach(ver ${flags})
      set(CMAKE_CUDA_ARCHITECTURES "${ver}-real;${ver}-virtual;${CMAKE_CUDA_ARCHITECTURES}")
    endforeach()
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
  find_package(OpenMP REQUIRED)
  target_link_libraries(${target} PUBLIC OpenMP::OpenMP_CXX)

  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:${GEN_CODE}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)

  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")
    set_property(TARGET ${target} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
  endif (CMAKE_VERSION VERSION_GREATER_EQUAL "3.18")

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

  target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_CUDA=1 -DTHRUST_IGNORE_CUB_VERSION_CHECK=1)
  target_include_directories(${target} PRIVATE ${xgboost_SOURCE_DIR}/cub/)

  if (MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
  endif (MSVC)

  set_target_properties(${target} PROPERTIES
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF)

  if (HIDE_CXX_SYMBOLS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fvisibility=hidden>)
  endif (HIDE_CXX_SYMBOLS)

  if (USE_NCCL)
    find_package(Nccl REQUIRED)
    target_include_directories(${target} PRIVATE ${NCCL_INCLUDE_DIR})
    target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_NCCL=1)
    target_link_libraries(${target} PUBLIC ${NCCL_LIBRARY})
  endif (USE_NCCL)
endfunction(xgboost_set_cuda_flags)
