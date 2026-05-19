cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# scikit-build-core entry point for the XGBoost Python wheel.
#
# Three build modes are supported, mirroring what the old custom
# PEP 517 backend (python-package/packager/) did in Python:
#
#   1. Reuse a prebuilt libxgboost.{so,dylib,dll} that already lives in
#      ${XGBOOST_PREBUILT_LIB_DIR} (default: ${C++ source}/lib). This is
#      the dominant CI path: CI builds libxgboost via CMake at the repo
#      root, drops it in lib/, then this CMakeLists.txt just packages it.
#
#   2. Build libxgboost from source by add_subdirectory()-ing the C++
#      tree. Used by `pip install python-package/` when no prebuilt lib
#      is available, and by sdist-driven installs (where the C++ tree
#      lives in cpp_src/ next to this file, staged by prepare_sdist.py).
#
#   3. Reuse a system-installed libxgboost by setting
#      -DXGBOOST_USE_SYSTEM_LIBXGBOOST=ON. In that mode we don't install
#      any shared library into the wheel; the runtime loader in
#      xgboost/libpath.py falls back to sys.base_prefix/lib.
#
# Starting in LANGUAGES NONE so CMake doesn't probe for a compiler when
# we end up in modes 1 or 3 (the common cases).
project(xgboost_python LANGUAGES NONE)

# ---------------------------------------------------------------------------
# Resolve the C++ source location.
# ---------------------------------------------------------------------------
#  - sdist install:  cpp_src/ next to this CMakeLists.txt (staged by
#                    ops/script/prepare_sdist.py before `python -m build --sdist`).
#  - in-repo build:  parent directory of python-package/ (the repo root).
if(EXISTS "${PROJECT_SOURCE_DIR}/cpp_src/CMakeLists.txt")
  set(_XGB_CPP_SRC "${PROJECT_SOURCE_DIR}/cpp_src")
else()
  set(_XGB_CPP_SRC "${PROJECT_SOURCE_DIR}/..")
endif()
get_filename_component(_XGB_CPP_SRC "${_XGB_CPP_SRC}" ABSOLUTE)

# Platform-specific library file name (matches xgboost/libpath.py).
if(WIN32)
  set(_XGB_LIBNAME "xgboost.dll")
elseif(APPLE)
  set(_XGB_LIBNAME "libxgboost.dylib")
else()
  set(_XGB_LIBNAME "libxgboost.so")
endif()

option(XGBOOST_USE_SYSTEM_LIBXGBOOST
  "Don't bundle libxgboost into the wheel; the runtime loader will find it on the system" OFF)
set(XGBOOST_PREBUILT_LIB_DIR "${_XGB_CPP_SRC}/lib" CACHE PATH
  "Directory holding a prebuilt libxgboost to bundle into the wheel")

if(XGBOOST_USE_SYSTEM_LIBXGBOOST)
  message(STATUS
    "xgboost-python: XGBOOST_USE_SYSTEM_LIBXGBOOST=ON — not bundling libxgboost; "
    "runtime will fall back to sys.base_prefix/lib")
  return()
endif()

if(EXISTS "${XGBOOST_PREBUILT_LIB_DIR}/${_XGB_LIBNAME}")
  # If lib/libxgboost.so is a symlink (the usual layout when CMake has just
  # built libxgboost with SOVERSION-based linking: .so -> .so.3 -> .so.3.3.0),
  # resolve it to the underlying file before install. install(FILES <symlink>)
  # would otherwise preserve the symlink itself, which becomes a dangling link
  # in the wheel staging dir and gets silently dropped by the wheel packer.
  file(REAL_PATH "${XGBOOST_PREBUILT_LIB_DIR}/${_XGB_LIBNAME}" _XGB_LIB_RESOLVED)
  message(STATUS
    "xgboost-python: bundling prebuilt ${_XGB_LIB_RESOLVED} into the wheel "
    "as xgboost/lib/${_XGB_LIBNAME}")
  install(FILES "${_XGB_LIB_RESOLVED}"
          DESTINATION xgboost/lib
          RENAME "${_XGB_LIBNAME}"
          COMPONENT XGBoostPython)
else()
  message(STATUS "xgboost-python: building libxgboost from sources at ${_XGB_CPP_SRC}")
  # Suppress the top-level CMakeLists.txt's default install() rules — they
  # are for system installs (headers, xgboost-config.cmake, etc.) and would
  # otherwise leak into the wheel. We install only the shared library target,
  # below, to its wheel-specific destination.
  set(XGBOOST_FOR_PYTHON_WHEEL ON CACHE BOOL "" FORCE)
  set(KEEP_BUILD_ARTIFACTS_IN_BINARY_DIR ON CACHE BOOL "" FORCE)
  add_subdirectory("${_XGB_CPP_SRC}" xgboost_cpp_build)
  # Install the resolved target file as the unversioned library name used by
  # xgboost/libpath.py. This mirrors the prebuilt-library path above and avoids
  # putting versioned-library symlinks into the wheel.
  install(FILES "$<TARGET_FILE:xgboost>"
          DESTINATION xgboost/lib
          RENAME "${_XGB_LIBNAME}"
          COMPONENT XGBoostPython)
endif()
