#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tries to find NCCL headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(Nccl)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  NCCL_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the NCCL installation.
#              The environment variable NCCL_ROOT overrides this variable.
#  NCCL_INCLUDE_DIR - Directory containing nccl.h.
#  NCCL_LIBRARY - Full path to the NCCL library.
#
# This module defines
#  Nccl_FOUND, whether nccl has been found
#  NCCL_INCLUDE_DIR, directory containing header
#  NCCL_LIBRARY, path to nccl library
#  NCCL_LIB_NAME, nccl library name
#  nccl::nccl, imported target for NCCL
#
# This module assumes that the user has already called find_package(CUDA)

if(BUILD_WITH_SHARED_NCCL)
  # libnccl.so
  set(NCCL_LIB_NAME nccl)
else()
  # libnccl_static.a
  set(NCCL_LIB_NAME nccl_static)
endif()

set(_nccl_roots)
if(DEFINED ENV{NCCL_ROOT})
  list(APPEND _nccl_roots "$ENV{NCCL_ROOT}")
endif()
if(NCCL_ROOT)
  list(APPEND _nccl_roots "${NCCL_ROOT}")
endif()
set(_nccl_library_hints)
set(_nccl_user_library)
set(_nccl_auto_library OFF)
if(NCCL_LIBRARY)
  if(NCCL_LIBRARY_AUTO_FOUND AND "${NCCL_LIBRARY}" STREQUAL "${NCCL_LIBRARY_AUTO_FOUND}")
    set(_nccl_auto_library ON)
  elseif(IS_DIRECTORY "${NCCL_LIBRARY}")
    message(FATAL_ERROR
      "NCCL_LIBRARY must be the full path to the NCCL library file, "
      "not a directory: ${NCCL_LIBRARY}. Use NCCL_ROOT to specify an "
      "installation prefix, or NCCL_INCLUDE_DIR and NCCL_LIBRARY to "
      "specify the header directory and library file separately.")
  else()
    set(_nccl_user_library "${NCCL_LIBRARY}")
  endif()
endif()

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS ${_nccl_roots}
  PATH_SUFFIXES include)

if(_nccl_user_library)
  set(NCCL_LIBRARY "${_nccl_user_library}" CACHE FILEPATH "Path to NCCL library" FORCE)
elseif(NOT USE_DLOPEN_NCCL)
  unset(NCCL_LIBRARY CACHE)
  find_library(NCCL_LIBRARY
    NAMES ${NCCL_LIB_NAME}
    HINTS ${_nccl_library_hints} ${_nccl_roots}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu)
  if(NCCL_LIBRARY)
    set(NCCL_LIBRARY_AUTO_FOUND "${NCCL_LIBRARY}" CACHE INTERNAL
      "NCCL library found by FindNccl")
  endif()
elseif(_nccl_auto_library)
  unset(NCCL_LIBRARY CACHE)
  unset(NCCL_LIBRARY_AUTO_FOUND CACHE)
endif()

if(USE_DLOPEN_NCCL)
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Nccl DEFAULT_MSG NCCL_INCLUDE_DIR)

  mark_as_advanced(NCCL_INCLUDE_DIR)
else()
  message(STATUS "Using nccl library: ${NCCL_LIBRARY}")

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Nccl DEFAULT_MSG
    NCCL_INCLUDE_DIR NCCL_LIBRARY)

  mark_as_advanced(
    NCCL_INCLUDE_DIR
    NCCL_LIBRARY
  )
endif()

if(Nccl_FOUND AND NOT TARGET nccl::nccl)
  if(USE_DLOPEN_NCCL)
    add_library(nccl::nccl INTERFACE IMPORTED)
  else()
    add_library(nccl::nccl UNKNOWN IMPORTED)
    set_target_properties(nccl::nccl PROPERTIES IMPORTED_LOCATION "${NCCL_LIBRARY}")
  endif()

  set_target_properties(nccl::nccl PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NCCL_INCLUDE_DIR}")
endif()
