export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -O3 -msse2 -fopenmp

# specify tensor path
BIN = xgboost
OBJ = xgboost.o
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm 

xgboost.o: booster/xgboost.h booster/xgboost_data.h booster/xgboost.cpp booster/*/*.hpp booster/*/*.h
xgboost: regression/xgboost_reg_main.cpp regression/*.h xgboost.o

$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
