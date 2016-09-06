# - Try to find the Crypto libcrypto library
# Once done this will define
#
#  CRYPTO_FOUND - system has the Crypto libcrypto library
#  CRYPTO_INCLUDE_DIR - the Crypto libcrypto include directory
#  CRYPTO_LIBRARIES - The libraries needed to use Crypto libcrypto

# Copyright (c) 2009, Matteo Panella, <morpheus@azzurra.org>
# Copyright (c) 2006, Alexander Neundorf, <neundorf@kde.org>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.


IF(CRYPTO_LIBRARIES)
   SET(Crypto_FIND_QUIETLY TRUE)
ENDIF(CRYPTO_LIBRARIES)

IF(SSL_EAY_DEBUG AND SSL_EAY_RELEASE)
   SET(LIB_FOUND 1)
ENDIF(SSL_EAY_DEBUG AND SSL_EAY_RELEASE)

FIND_PATH(CRYPTO_INCLUDE_DIR openssl/crypto.h )
FIND_LIBRARY(CRYPTO_LIBRARIES NAMES crypto )

IF(CRYPTO_INCLUDE_DIR AND CRYPTO_LIBRARIES)
   SET(CRYPTO_FOUND TRUE)
ELSE(CRYPTO_INCLUDE_DIR AND CRYPTO_LIBRARIES)
   SET(CRYPTO_FOUND FALSE)
ENDIF (CRYPTO_INCLUDE_DIR AND CRYPTO_LIBRARIES)

IF (CRYPTO_FOUND)
   IF (NOT Crypto_FIND_QUIETLY)
      MESSAGE(STATUS "Found libcrypto: ${CRYPTO_LIBRARIES}")
   ENDIF (NOT Crypto_FIND_QUIETLY)
ELSE (CRYPTO_FOUND)
   IF (Crypto_FIND_REQUIRED)
      MESSAGE(FATAL_ERROR "Could NOT find libcrypto")
   ENDIF (Crypto_FIND_REQUIRED)
ENDIF (CRYPTO_FOUND)

MARK_AS_ADVANCED(CRYPTO_INCLUDE_DIR CRYPTO_LIBRARIES)