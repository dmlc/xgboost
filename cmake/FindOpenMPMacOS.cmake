# Find OpenMP library on MacOS
# Automatically handle locating libomp from the Homebrew package manager

# lint_cmake: -package/consistency

macro(find_openmp_macos)
  if(NOT APPLE)
    message(FATAL_ERROR "${CMAKE_CURRENT_FUNCTION}() must only be used on MacOS")
  endif()
  find_package(OpenMP)
  if(NOT OpenMP_FOUND)
    # Try again with extra path info. This step is required for libomp 15+ from Homebrew,
    # as libomp 15.0+ from brew is keg-only
    # See https://github.com/Homebrew/homebrew-core/issues/112107#issuecomment-1278042927.
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(OpenMP_C_FLAGS
      "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_FLAGS
      "-Xpreprocessor -fopenmp -I${HOMEBREW_LIBOMP_PREFIX}/include")
    set(OpenMP_C_LIB_NAMES omp)
    set(OpenMP_CXX_LIB_NAMES omp)
    set(OpenMP_omp_LIBRARY ${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib)
    find_package(OpenMP REQUIRED)
  endif()
endmacro()

# Patch an XGBoost shared library so that it depends on @rpath/libomp.dylib
# instead of /opt/homebrew/opt/libomp/lib/libomp.dylib or other hard-coded paths.
# Doing so enables XGBoost to interoperate with multiple kinds of OpenMP
# libraries. See https://github.com/lightgbm-org/LightGBM/pull/6391 for detailed
# explanation. Adapted from https://github.com/lightgbm-org/LightGBM/pull/6391
# by James Lamb.
# MacOS only.
function(patch_openmp_path_macos target)
  if(NOT APPLE)
    message(FATAL_ERROR "${CMAKE_CURRENT_FUNCTION}() must only be used on MacOS")
  endif()
  # Get path to libomp found at build time
  get_target_property(
    __OpenMP_LIBRARY_LOCATION
    OpenMP::OpenMP_CXX
    INTERFACE_LINK_LIBRARIES
  )
  # Get the base name of the OpenMP lib
  # Usually: libomp.dylib, libgomp.dylib, or libiomp.dylib
  get_filename_component(
    __OpenMP_LIBRARY_NAME
    ${__OpenMP_LIBRARY_LOCATION}
    NAME
  )
  # Get the directory containing the OpenMP lib
  get_filename_component(
    __OpenMP_LIBRARY_DIR
    ${__OpenMP_LIBRARY_LOCATION}
    DIRECTORY
  )
  # Override the absolute path to OpenMP with a relative one using @rpath.
  #
  # This also ensures that if a libomp.dylib has already been loaded, it'll just use that.
  add_custom_command(
    TARGET ${target}
    POST_BUILD
      COMMAND
        install_name_tool
        -change
        ${__OpenMP_LIBRARY_LOCATION}
        "@rpath/${__OpenMP_LIBRARY_NAME}"
        "$<TARGET_FILE:${target}>"
      VERBATIM
  )
  message(STATUS
    "${target}: "
    "Replacing hard-coded OpenMP install_name with '@rpath/${__OpenMP_LIBRARY_NAME}'..."
  )
  # Add RPATH entries to ensure the loader looks in the following locations:
  #
  #   - R builds: wherever the active R toolchain's OpenMP discovery found libomp.
  #   - Other builds: Homebrew's libomp followed by the discovered OpenMP library directory.
  #
  # Note: This list will only be used if libomp.dylib isn't already loaded into memory.
  #       So Conda users will likely use ${CONDA_PREFIX}/libomp.dylib
  if(R_LIB)
    set(__OPENMP_RPATH "${__OpenMP_LIBRARY_DIR}")
  else()
    execute_process(COMMAND brew --prefix libomp
                    OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(__OPENMP_RPATH "${HOMEBREW_LIBOMP_PREFIX}/lib;${__OpenMP_LIBRARY_DIR}")
  endif()
  set_target_properties(
    ${target}
    PROPERTIES
      BUILD_WITH_INSTALL_RPATH TRUE
      INSTALL_RPATH "${__OPENMP_RPATH}"
      INSTALL_RPATH_USE_LINK_PATH FALSE
  )
endfunction()
