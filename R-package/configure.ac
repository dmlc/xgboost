### configure.ac					-*- Autoconf -*-

AC_PREREQ(2.62)

AC_INIT([xgboost],[0.6-3],[],[xgboost],[])

OPENMP_CXXFLAGS=""

if test `uname -s` = "Linux"
then
  OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CFLAGS)"
fi

if test `uname -s` = "Darwin"
then
  OPENMP_CXXFLAGS="\$(SHLIB_OPENMP_CFLAGS)"
fi

AC_SUBST(OPENMP_CXXFLAGS)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT

