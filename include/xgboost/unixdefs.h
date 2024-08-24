#if defined(UNIXDEFS_H) && !defined(_MSC_VER) && \
    !defined(__MINGW32__) && !defined(__CYGWIN__) && \
    !defined(_WIN32) && !defined(__WINDOWS__) && \
    !defined(__MUSL__)
#define UNIXDEFS_H

#define _GNU_SOURCE
#include <features.h>
#ifndef __USE_GNU
    #define __MUSL__
#endif
#undef _GNU_SOURCE

#endif
