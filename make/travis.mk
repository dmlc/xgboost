
# the additional link flags you want to add
ADD_LDFLAGS =

# the additional compile flags you want to add
ADD_CFLAGS =

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
LIB_RABIT = librabit.a

# path to libjvm.so
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server

# path to googletest and whether to measure coverage or not
GTEST_PATH =
WITH_COVER = 1

# List of additional plugins, checkout plugin folder.
# uncomment the following lines to include these plugins
# you can also add your own plugin like this
#
XGB_PLUGINS += plugin/example/plugin.mk
XGB_PLUGINS += plugin/lz4/plugin.mk
XGB_PLUGINS += plugin/dense_parser/plugin.mk
