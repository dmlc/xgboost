
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
  endif()
endfunction(msvc_use_static_runtime)

# Set output directory of target, ignoring debug or release
function(set_output_directory target dir)
	set_target_properties(${target} PROPERTIES 
		RUNTIME_OUTPUT_DIRECTORY ${dir}
		RUNTIME_OUTPUT_DIRECTORY_DEBUG ${dir}
		RUNTIME_OUTPUT_DIRECTORY_RELEASE ${dir}
		LIBRARY_OUTPUT_DIRECTORY ${dir} 
		LIBRARY_OUTPUT_DIRECTORY_DEBUG ${dir} 
		LIBRARY_OUTPUT_DIRECTORY_RELEASE ${dir} 
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
  # Set up architecture flags
  if(NOT flags) 
    if((CUDA_VERSION_MAJOR EQUAL 9) OR (CUDA_VERSION_MAJOR GREATER 9))
      set(flags "35;50;52;60;61;70")
    else()
      set(flags "35;50;52;60;61")
    endif()
  endif()
  # Generate SASS
  foreach(ver ${flags})
    set(${out} "${${out}}-gencode arch=compute_${ver},code=sm_${ver};")
  endforeach()
  # Generate PTX for last architecture
  list(GET flags -1 ver)
  set(${out} "${${out}}-gencode arch=compute_${ver},code=compute_${ver};")
  
  set(${out} "${${out}}" PARENT_SCOPE)
endfunction(format_gencode_flags flags)

# Assembles the R-package files in build_dir;
# if necessary, installs the main R package dependencies;
# runs R CMD INSTALL.
function(setup_rpackage_install_target rlib_target build_dir)
  install(CODE "file(REMOVE_RECURSE \"${build_dir}/R-package\")")
  install(
    DIRECTORY "${PROJECT_SOURCE_DIR}/R-package"
    DESTINATION "${build_dir}"
    REGEX "src/*" EXCLUDE
    REGEX "R-package/configure" EXCLUDE
  )
  install(TARGETS ${rlib_target}
    LIBRARY DESTINATION "${build_dir}/R-package/src/"
    RUNTIME DESTINATION "${build_dir}/R-package/src/")
  install(CODE "file(WRITE \"${build_dir}/R-package/src/Makevars\" \"all:\")")
  install(CODE "file(WRITE \"${build_dir}/R-package/src/Makevars.win\" \"all:\")")
  set(XGB_DEPS_SCRIPT
    "deps = setdiff(c('data.table', 'magrittr', 'stringi'), rownames(installed.packages()));\
    if(length(deps)>0) install.packages(deps, repo = 'https://cloud.r-project.org/')")
  install(CODE "execute_process(COMMAND \"${LIBR_EXECUTABLE}\" \"-q\" \"-e\" \"${XGB_DEPS_SCRIPT}\")")
  install(CODE "execute_process(COMMAND \"${LIBR_EXECUTABLE}\" CMD INSTALL\
    \"--no-multiarch\" \"--build\" \"${build_dir}/R-package\")")
endfunction(setup_rpackage_install_target)
