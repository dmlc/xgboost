# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# - Find Arrow Python (arrow/python/api.h, libarrow_python.a, libarrow_python.so)
#
# This module requires Arrow from which it uses
#  arrow_find_package()
#
# This module defines
#  ARROW_PYTHON_FOUND, whether Arrow Python has been found
#  ARROW_PYTHON_IMPORT_LIB,
#    path to libarrow_python's import library (Windows only)
#  ARROW_PYTHON_INCLUDE_DIR, directory containing headers
#  ARROW_PYTHON_LIB_DIR, directory containing Arrow Python libraries
#  ARROW_PYTHON_SHARED_LIB, path to libarrow_python's shared library
#  ARROW_PYTHON_STATIC_LIB, path to libarrow_python.a

if(DEFINED ARROW_PYTHON_FOUND)
  return()
endif()

set(find_package_arguments)
if(${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION)
  list(APPEND find_package_arguments "${${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION}")
endif()
if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
  list(APPEND find_package_arguments REQUIRED)
endif()
if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
  list(APPEND find_package_arguments QUIET)
endif()
find_package(Arrow ${find_package_arguments})

if(ARROW_FOUND)
  arrow_find_package(ARROW_PYTHON
                     "${ARROW_HOME}"
                     arrow_python
                     arrow/python/api.h
                     ArrowPython
                     arrow-python)
  if(NOT ARROW_PYTHON_VERSION)
    set(ARROW_PYTHON_VERSION "${ARROW_VERSION}")
  endif()
endif()

if("${ARROW_PYTHON_VERSION}" VERSION_EQUAL "${ARROW_VERSION}")
  set(ARROW_PYTHON_VERSION_MATCH TRUE)
else()
  set(ARROW_PYTHON_VERSION_MATCH FALSE)
endif()

mark_as_advanced(ARROW_PYTHON_IMPORT_LIB
                 ARROW_PYTHON_INCLUDE_DIR
                 ARROW_PYTHON_LIBS
                 ARROW_PYTHON_LIB_DIR
                 ARROW_PYTHON_SHARED_IMP_LIB
                 ARROW_PYTHON_SHARED_LIB
                 ARROW_PYTHON_STATIC_LIB
                 ARROW_PYTHON_VERSION
                 ARROW_PYTHON_VERSION_MATCH)

find_package_handle_standard_args(ArrowPython
                                  REQUIRED_VARS
                                  ARROW_PYTHON_INCLUDE_DIR
                                  ARROW_PYTHON_LIB_DIR
                                  ARROW_PYTHON_VERSION_MATCH
                                  VERSION_VAR
                                  ARROW_PYTHON_VERSION)
set(ARROW_PYTHON_FOUND ${ArrowPython_FOUND})

if(ArrowPython_FOUND AND NOT ArrowPython_FIND_QUIETLY)
  message(STATUS "Found the Arrow Python by ${ARROW_PYTHON_FIND_APPROACH}")
  message(STATUS "Found the Arrow Python shared library: ${ARROW_PYTHON_SHARED_LIB}")
  message(STATUS "Found the Arrow Python import library: ${ARROW_PYTHON_IMPORT_LIB}")
  message(STATUS "Found the Arrow Python static library: ${ARROW_PYTHON_STATIC_LIB}")
endif()
