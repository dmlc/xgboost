# CMake module for R
# Borrows ideas from RStudio's FindLibR.cmake
#
# Defines the following:
#  LIBR_FOUND
#  LIBR_HOME
#  LIBR_EXECUTABLE
#  LIBR_INCLUDE_DIRS
#  LIBR_LIB_DIR
#  LIBR_CORE_LIBRARY
# and a cmake function to create R.lib for MSVC
#
# The following could be provided by user through cmake's -D options:
#  LIBR_EXECUTABLE (for unix and win)
#  R_VERSION (for win)
#  R_ARCH (for win 64 when want 32 bit build)
#
# TODO: 
# - someone to verify OSX detection, 
# - possibly, add OSX detection based on current R in PATH or LIBR_EXECUTABLE
# - improve registry-based R_HOME detection in Windows (from a set of R_VERSION's)


# Windows users might want to change this to their R version:
if(NOT R_VERSION)
  set(R_VERSION "3.4.1")
endif()
if(NOT R_ARCH)
  if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "4")
    set(R_ARCH "i386")
  else()
    set(R_ARCH "x64")
  endif()
endif()


# Creates R.lib and R.def in the build directory for linking with MSVC
function(create_rlib_for_msvc)
  # various checks and warnings
  if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "create_rlib_for_msvc() can only be used with MSVC")
  endif()
  if(NOT EXISTS "${LIBR_LIB_DIR}")
    message(FATAL_ERROR "LIBR_LIB_DIR was not set!")
  endif()
  find_program(GENDEF_EXE gendef)
  find_program(DLLTOOL_EXE dlltool)
  if(NOT GENDEF_EXE OR NOT DLLTOOL_EXE)
    message(FATAL_ERROR "\nEither gendef.exe or dlltool.exe not found!\
      \nDo you have Rtools installed with its MinGW's bin/ in PATH?")
  endif()  
  # extract symbols from R.dll into R.def and R.lib import library
  execute_process(COMMAND gendef
    "-" "${LIBR_LIB_DIR}/R.dll"
    OUTPUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/R.def")
  execute_process(COMMAND dlltool
    "--input-def" "${CMAKE_CURRENT_BINARY_DIR}/R.def"
    "--output-lib" "${CMAKE_CURRENT_BINARY_DIR}/R.lib")
endfunction(create_rlib_for_msvc)


# detection for OSX
if(APPLE)

  find_library(LIBR_LIBRARIES R)

  if(LIBR_LIBRARIES MATCHES ".*\\.framework")
    set(LIBR_HOME "${LIBR_LIBRARIES}/Resources" CACHE PATH "R home directory")
    set(LIBR_INCLUDE_DIRS "${LIBR_HOME}/include" CACHE PATH "R include directory")
    set(LIBR_EXECUTABLE "${LIBR_HOME}/R" CACHE PATH "R executable")
    set(LIBR_LIB_DIR "${LIBR_HOME}/lib" CACHE PATH "R lib directory")
  else()
    get_filename_component(_LIBR_LIBRARIES "${LIBR_LIBRARIES}" REALPATH)
    get_filename_component(_LIBR_LIBRARIES_DIR "${_LIBR_LIBRARIES}" DIRECTORY)
    set(LIBR_EXECUTABLE "${_LIBR_LIBRARIES_DIR}/../bin/R")
    execute_process(
      COMMAND ${LIBR_EXECUTABLE} "--slave" "--vanilla" "-e" "cat(R.home())"
      OUTPUT_VARIABLE LIBR_HOME)
    set(LIBR_HOME ${LIBR_HOME} CACHE PATH "R home directory")
    set(LIBR_INCLUDE_DIRS "${LIBR_HOME}/include" CACHE PATH "R include directory")
    set(LIBR_LIB_DIR "${LIBR_HOME}/lib" CACHE PATH "R lib directory")
  endif()
  
# detection for UNIX & Win32
else()

  # attempt to find R executable
  if(NOT LIBR_EXECUTABLE)
    find_program(LIBR_EXECUTABLE NAMES R R.exe)
  endif()
  
  if(UNIX)

    if(NOT LIBR_EXECUTABLE)
      message(FATAL_ERROR "Unable to locate R executable.\
        \nEither add its location to PATH or provide it through the LIBR_EXECUTABLE cmake variable")
    endif()

    # ask R for the home path
    execute_process(
      COMMAND ${LIBR_EXECUTABLE} "--slave" "--vanilla" "-e" "cat(R.home())"
      OUTPUT_VARIABLE LIBR_HOME
    )
    # ask R for the include dir
    execute_process(
      COMMAND ${LIBR_EXECUTABLE} "--slave" "--no-save" "-e" "cat(R.home('include'))"
      OUTPUT_VARIABLE LIBR_INCLUDE_DIRS
    )
    # ask R for the lib dir
    execute_process(
      COMMAND ${LIBR_EXECUTABLE} "--slave" "--no-save" "-e" "cat(R.home('lib'))"
      OUTPUT_VARIABLE LIBR_LIB_DIR
    )

  # Windows
  else()
    # ask R for R_HOME 
    if(LIBR_EXECUTABLE)
      execute_process(
        COMMAND ${LIBR_EXECUTABLE} "--slave" "--no-save" "-e" "cat(normalizePath(R.home(),winslash='/'))"
        OUTPUT_VARIABLE LIBR_HOME)
    endif()
    # if R executable not available, query R_HOME path from registry
    if(NOT LIBR_HOME)
      get_filename_component(LIBR_HOME
        "[HKEY_LOCAL_MACHINE\\SOFTWARE\\R-core\\R\\${R_VERSION};InstallPath]"
        ABSOLUTE)
      if(NOT LIBR_HOME)
        message(FATAL_ERROR "\nUnable to locate R executable.\
          \nEither add its location to PATH or provide it through the LIBR_EXECUTABLE cmake variable")
      endif()
    endif()
    # set exe location based on R_ARCH
    if(NOT LIBR_EXECUTABLE)
      set(LIBR_EXECUTABLE "${LIBR_HOME}/bin/${R_ARCH}/R.exe")
    endif()
    # set other R paths based on home path
    set(LIBR_INCLUDE_DIRS "${LIBR_HOME}/include")
    set(LIBR_LIB_DIR "${LIBR_HOME}/bin/${R_ARCH}")
 
message(STATUS "LIBR_HOME [${LIBR_HOME}]")
message(STATUS "LIBR_EXECUTABLE [${LIBR_EXECUTABLE}]")
message(STATUS "LIBR_INCLUDE_DIRS [${LIBR_INCLUDE_DIRS}]")
message(STATUS "LIBR_LIB_DIR [${LIBR_LIB_DIR}]")
message(STATUS "LIBR_CORE_LIBRARY [${LIBR_CORE_LIBRARY}]")

  endif()

endif()

if(WIN32 AND MSVC)
  # create a local R.lib import library for R.dll if it doesn't exist
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/R.lib")
    create_rlib_for_msvc()
  endif()
endif()

# look for the core R library
find_library(LIBR_CORE_LIBRARY NAMES R
  HINTS "${CMAKE_CURRENT_BINARY_DIR}" "${LIBR_LIB_DIR}" "${LIBR_HOME}/bin" "${LIBR_LIBRARIES}")
if(LIBR_CORE_LIBRARY-NOTFOUND)
  message(STATUS "Could not find R core shared library.")
endif()

set(LIBR_HOME ${LIBR_HOME} CACHE PATH "R home directory")
set(LIBR_EXECUTABLE ${LIBR_EXECUTABLE} CACHE PATH "R executable")
set(LIBR_INCLUDE_DIRS ${LIBR_INCLUDE_DIRS} CACHE PATH "R include directory")
set(LIBR_LIB_DIR ${LIBR_LIB_DIR} CACHE PATH "R shared libraries directory")
set(LIBR_CORE_LIBRARY ${LIBR_CORE_LIBRARY} CACHE PATH "R core shared library")

# define find requirements
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibR DEFAULT_MSG
  LIBR_HOME
  LIBR_EXECUTABLE
  LIBR_INCLUDE_DIRS
  LIBR_LIB_DIR
  LIBR_CORE_LIBRARY
)

if(LIBR_FOUND)
  message(STATUS "Found R: ${LIBR_EXECUTABLE}")
endif()
