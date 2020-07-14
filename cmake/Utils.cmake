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
		ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${dir}
	)
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
    if(CUDA_VERSION VERSION_GREATER_EQUAL "10.0")
      set(flags "35;50;52;60;61;70;75")
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL "9.0")
      set(flags "35;50;52;60;61;70")
    else()
      set(flags "35;50;52;60;61")
    endif()
  endif()
  # Generate SASS
  foreach(ver ${flags})
    set(${out} "${${out}}--generate-code=arch=compute_${ver},code=sm_${ver};")
  endforeach()
  # Generate PTX for last architecture
  list(GET flags -1 ver)
  set(${out} "${${out}}--generate-code=arch=compute_${ver},code=compute_${ver};")

  set(${out} "${${out}}" PARENT_SCOPE)
endfunction(format_gencode_flags flags)

macro(enable_nvtx target)
  find_package(NVTX REQUIRED)
  target_include_directories(${target} PRIVATE "${NVTX_INCLUDE_DIR}")
  target_link_libraries(${target} PRIVATE "${NVTX_LIBRARY}")
  target_compile_definitions(${target} PRIVATE -DXGBOOST_USE_NVTX=1)
endmacro()

macro(enable_arrow_if_available target)
  find_package(Arrow)
  find_package(ArrowPython)
  if (ARROW_FOUND AND ARROW_PYTHON_FOUND)
    find_package(Python3 COMPONENTS Development REQUIRED)
    target_include_directories(${target}  PRIVATE
                                ${ARROW_INCLUDE_DIR}
                                ${ARROW_PYTHON_INCLUDE_DIR}
                                ${Python3_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE 
                          ${ARROW_SHARED_LIB}
                          ${ARROW_PYTHON_SHARED_LIB}
                          ${Python3_LIBRARIES})
    target_compile_definitions(${target} PRIVATE
                      -DXGBOOST_BUILD_ARROW_SUPPORT=1
                      -D_GLIBCXX_USE_CXX11_ABI=0)
  endif (ARROW_FOUND AND ARROW_PYTHON_FOUND)
endmacro()
