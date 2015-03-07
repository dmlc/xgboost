# specify tensor path
BIN = kmeans.rabit 
MOCKBIN= kmeans.mock
MPIBIN = kmeans.mpi
# objectives that makes up rabit library
OBJ = kmeans.o

# common build script for programs
include ../make/common.mk

# dependenies here
kmeans.rabit: kmeans.o lib
kmeans.mock: kmeans.o lib
kmeans.mpi: kmeans.o libmpi
kmeans.o: kmeans.cc ../../src/*.h
