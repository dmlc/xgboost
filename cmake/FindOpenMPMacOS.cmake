function(find_openmp_macos)
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
endfunction()
