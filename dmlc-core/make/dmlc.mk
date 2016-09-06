#---------------------------------------------------------------------------------------
#  mshadow configuration script
#
#  include dmlc.mk after the variables are set
#
#  Add DMLC_CFLAGS to the compile flags
#  Add DMLC_LDFLAGS to the linker flags
#----------------------------------------------------------------------------------------
ifndef LIBJVM
	LIBJVM=$(JAVA_HOME)/jre/lib/amd64/server
endif

ifneq ($(USE_OPENMP), 0)
	DMLC_CFLAGS += -fopenmp
	DMLC_LDFLAGS += -fopenmp
endif

# Mac OS X does not support "-lrt" flag
ifeq ($(OS), Windows_NT)
	UNAME=Windows
else 
	UNAME=$(shell uname)
endif

ifeq ($(UNAME), Linux)
    DMLC_LDFLAGS += -lrt
endif

# handle fpic options
ifndef WITH_FPIC
	WITH_FPIC = 1
endif

ifeq ($(WITH_FPIC), 1)
	DMLC_CFLAGS += -fPIC
endif

# Using default hadoop_home
ifndef HADOOP_HDFS_HOME
	HADOOP_HDFS_HOME=$(HADOOP_HOME)
endif

ifeq ($(USE_HDFS),1)
	ifndef HDFS_INC_PATH
		HDFS_INC_PATH=$(HADOOP_HDFS_HOME)/include
	endif
	ifndef HDFS_LIB_PATH
		HDFS_LIB_PATH=$(HADOOP_HDFS_HOME)/lib/native
	endif

	DMLC_CFLAGS+= -DDMLC_USE_HDFS=1 -I$(HDFS_INC_PATH) -I$(JAVA_HOME)/include

	ifneq ("$(wildcard $(HDFS_LIB_PATH)/libhdfs.so)","")
		DMLC_LDFLAGS+= -L$(HDFS_LIB_PATH) -lhdfs
	else
		DMLC_LDFLAGS+= $(HDFS_LIB_PATH)/libhdfs.a
	endif
	DMLC_LDFLAGS += -L$(LIBJVM) -ljvm -Wl,-rpath=$(LIBJVM)
else
	DMLC_CFLAGS+= -DDMLC_USE_HDFS=0
endif

# setup S3
ifeq ($(USE_S3),1)
	DMLC_CFLAGS+= -DDMLC_USE_S3=1
	DMLC_LDFLAGS+= -lcurl -lssl -lcrypto
else
	DMLC_CFLAGS+= -DDMLC_USE_S3=0
endif

ifeq ($(USE_GLOG), 1)
	DMLC_CFLAGS += -DDMLC_USE_GLOG=1
	DMLC_LDFLAGS += -lglog
endif

ifeq ($(USE_AZURE),1)
	DMLC_CFLAGS+= -DDMLC_USE_AZURE=1
	DMLC_LDFLAGS+= -lazurestorage
else
	DMLC_CFLAGS+= -DDMLC_USE_AZURE=0
endif
