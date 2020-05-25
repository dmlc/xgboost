### configure.ac					-*- Autoconf -*-

AC_PREREQ(2.69)

AC_INIT([xgboost],[0.6-3],[],[xgboost],[])

# Use this line to set CC variable to a C compiler
AC_PROG_CC

### Check whether backtrace() is part of libc or the external lib libexecinfo
AC_MSG_CHECKING([Backtrace lib])
AC_MSG_RESULT([])
AC_CHECK_LIB([execinfo], [backtrace], [BACKTRACE_LIB=-lexecinfo], [BACKTRACE_LIB=''])

### Endian detection
AC_MSG_CHECKING([endian])
AC_MSG_RESULT([])
AC_RUN_IFELSE([AC_LANG_PROGRAM([[#include <stdint.h>]], [[const uint16_t endianness = 256; return !!(*(const uint8_t *)&endianness);]])],
  [ENDIAN_FLAG="-DDMLC_CMAKE_LITTLE_ENDIAN=1"],
  [ENDIAN_FLAG="-DDMLC_CMAKE_LITTLE_ENDIAN=0"])

OPENMP_CXXFLAGS=""

if test `uname -s` = "Linux"
then
  OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CXXFLAGS)"
fi

if test `uname -s` = "Darwin"
then
  OPENMP_CXXFLAGS='-Xclang -fopenmp'
  OPENMP_LIB='-lomp'
  ac_pkg_openmp=no
  AC_MSG_CHECKING([whether OpenMP will work in a package])
  AC_LANG_CONFTEST([AC_LANG_PROGRAM([[#include <omp.h>]], [[ return (omp_get_max_threads() <= 1); ]])])
  ${CC} -o conftest conftest.c ${OPENMP_LIB} ${OPENMP_CXXFLAGS} 2>/dev/null && ./conftest && ac_pkg_openmp=yes
  AC_MSG_RESULT([${ac_pkg_openmp}])
  if test "${ac_pkg_openmp}" = no; then
    OPENMP_CXXFLAGS=''
    OPENMP_LIB=''
    echo '*****************************************************************************************'
    echo 'WARNING: OpenMP is unavailable on this Mac OSX system. Training speed may be suboptimal.'
    echo '         To use all CPU cores for training jobs, you should install OpenMP by running\n'
    echo '             brew install libomp'
    echo '*****************************************************************************************'
  fi
fi

AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(OPENMP_LIB)
AC_SUBST(ENDIAN_FLAG)
AC_SUBST(BACKTRACE_LIB)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

