### configure.ac					-*- Autoconf -*-

AC_PREREQ(2.62)

AC_INIT([xgboost],[0.6-3],[],[xgboost],[])

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
  OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CXXFLAGS)"
  ac_pkg_openmp=no
  AC_MSG_CHECKING([whether OpenMP will work in a package])
  AC_LANG_CONFTEST(
  [AC_LANG_PROGRAM([[#include <omp.h>]], [[ return omp_get_num_threads (); ]])])
  PKG_CFLAGS="${OPENMP_CFLAGS}" PKG_LIBS="${OPENMP_CFLAGS}" "$RBIN" CMD SHLIB conftest.c 1>&AS_MESSAGE_LOG_FD 2>&AS_MESSAGE_LOG_FD && "$RBIN" --vanilla -q -e "dyn.load(paste('conftest',.Platform\$dynlib.ext,sep=''))" 1>&AS_MESSAGE_LOG_FD 2>&AS_MESSAGE_LOG_FD && ac_pkg_openmp=yes
  AC_MSG_RESULT([${ac_pkg_openmp}])
  if test "${ac_pkg_openmp}" = no; then
    OPENMP_CXXFLAGS=''
  fi
fi

AC_SUBST(OPENMP_CXXFLAGS)
AC_SUBST(ENDIAN_FLAG)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

