# this is the common build script for rabit programs
# you do not have to use it
export LDFLAGS= -L../../lib -pthread -lm -lrt
export CFLAGS = -Wall  -msse2  -Wno-unknown-pragmas -fPIC -I../../include  

# setup opencv
ifeq ($(USE_WORMHOLE),1)
	CFLAGS+= -DRABIT_USE_WORMHOLE=1 -I ../../wormhole/include
	LDFLAGS+= -L../../wormhole -lwormhole
else
	CFLAGS+= -DRABIT_USE_WORMHOLE=0
endif

# setup opencv
ifeq ($(USE_HDFS),1)
	CFLAGS+= -DRABIT_USE_HDFS=1 -I$(HADOOP_HDFS_HOME)/include -I$(JAVA_HOME)/include
	LDFLAGS+= -L$(HADOOP_HDFS_HOME)/lib/native -L$(LIBJVM) -lhdfs -ljvm
else
	CFLAGS+= -DRABIT_USE_HDFS=0
endif


.PHONY: clean all lib mpi

all: $(BIN) $(MOCKBIN)

mpi: $(MPIBIN)

lib:
	cd ../..;make lib/librabit.a lib/librabit_mock.a; cd -
libmpi:
	cd ../..;make lib/librabit_mpi.a;cd -


$(BIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) -lrabit $(LDFLAGS) 

$(MOCKBIN) : 
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc,  $^) -lrabit_mock $(LDFLAGS) 

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c %.cc, $^) )

$(MPIBIN) : 
	$(MPICXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a, $^)  $(LDFLAGS) -lrabit_mpi 

clean:
	$(RM) $(OBJ) $(BIN) $(MPIBIN) $(MOCKBIN) *~ ../src/*~
