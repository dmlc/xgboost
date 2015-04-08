#-----------------------------------------------------
#  rabit-learn: the configuration compile script
#
#  This is the default configuration setup for rabit-learn
#  If you want to change configuration, do the following steps:
#
#  - copy this file to the root of rabit-learn folder
#  - modify the configuration you want
#  - type make or make -j n on each of the folder
#----------------------------------------------------

# choice of compiler
export CC = gcc
export CXX = g++
export MPICXX = mpicxx

# whether use HDFS support during compile
USE_HDFS = 1

# whether use dmlc's io utils
USE_DMLC = 0

# path to libjvm.so
LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server
