export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -O3 -msse2  -Wno-unknown-pragmas -fopenmp 

# specify tensor path
BIN = xgboost
OBJ = 
.PHONY: clean all

all: $(BIN) $(OBJ)
export LDFLAGS= -pthread -lm 

xgboost: regrank/xgboost_regrank_main.cpp regrank/*.h regrank/*.hpp booster/*.h booster/*/*.hpp booster/*.hpp


$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) *~
