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

# - Find Arrow (arrow/api.h, libarrow.a, libarrow.so)
# - Find pyarrow (pyarrow.h, libarrow_python.so)
# This module defines
#  ARROW_FOUND, whether Arrow has been found
#  PYARROW_FOUND, whether Pyarrow has been found
#  ARROW_INCLUDE_DIR, directory containing headers
#  PYARROW_INCLUDE_DIR, directory containing headers
#  ARROW_SHARED_LIB, path to libarrow's shared library
#  PYARROW_SHARED_LIB, path to libarrow_python's shared library

set(ARROW_LIB_NAME arrow)
set(PYARROW_LIB_NAME arrow_python)

find_path(ARROW_INCLUDE_DIR
  NAMES arrow/api.h
  PATHS $ENV{ARROW_ROOT}/include ${ARROW_ROOT}/include)

find_path(PYARROW_INCLUDE_DIR
  NAMES arrow/python/pyarrow.h
  PATHS $ENV{ARROW_ROOT}/include ${ARROW_ROOT}/include)

find_library(ARROW_SHARED_LIB
  NAMES ${ARROW_LIB_NAME}
  PATHS $ENV{ARROW_ROOT}/lib ${ARROW_ROOT}/lib)

find_library(PYARROW_SHARED_LIB
  NAMES ${PYARROW_LIB_NAME}
  PATHS $ENV{ARROW_ROOT}/lib ${ARROW_ROOT}/lib)

message(STATUS "Using Arrow library: ${ARROW_SHARED_LIB}")
message(STATUS "Using Pyarrow library: ${PYARROW_SHARED_LIB}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Arrow DEFAULT_MSG
  ARROW_INCLUDE_DIR PYARROW_INCLUDE_DIR ARROW_SHARED_LIB PYARROW_SHARED_LIB)

mark_as_advanced(
  ARROW_INCLUDE_DIR
  PYARROW_INCLUDE_DIR
  ARROW_SHARED_LIB
  PYARROW_SHARED_LIB)
