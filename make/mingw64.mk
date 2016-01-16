#-----------------------------------------------------------
# xgboost: Configuration for MinGW(Windows 64bit)
# This allows to compile xgboost on windows by using mingw.
# You will need to get install an mingw toolchain.
# g++-4.6 or later is required.
#
# see config.mk for template.
#-----------------------------------------------------------
export CXX=g++ -m64
export CC=gcc -m64

# Whether enable openmp support, needed for multi-threading.
USE_OPENMP = 1

# whether use HDFS support during compile
USE_HDFS = 0

# whether use AWS S3 support during compile
USE_S3 = 0

# whether use Azure blob support during compile
USE_AZURE = 0

# Rabit library version,
# - librabit.a Normal distributed version.
# - librabit_empty.a Non distributed mock version,
LIB_RABIT = librabit_empty.a

DMLC_CFLAGS = -DDMLC_ENABLE_STD_THREAD=0
ADD_CFLAGS = -DDMLC_ENABLE_STD_THREAD=0