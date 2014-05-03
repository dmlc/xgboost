export CC  = gcc
export CXX = g++
export CFLAGS = -Wall -msse2  -Wno-unknown-pragmas -fopenmp

# specify tensor path
SLIB = libxgboostpy.so
.PHONY: clean all

all: $(SLIB)
export LDFLAGS= -pthread -lm 

libxgboostpy.so: xgboost_python.cpp ../regrank/*.h ../booster/*.h ../booster/*/*.hpp ../booster/*.hpp

$(SLIB) :
	$(CXX) $(CFLAGS) -fPIC $(LDFLAGS) -shared -o $@ $(filter %.cpp %.o %.c, $^)
$(BIN) : 
	$(CXX) $(CFLAGS) $(LDFLAGS) -o $@ $(filter %.cpp %.o %.c, $^)

$(OBJ) : 
	$(CXX) -c $(CFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

install:
	cp -f -r $(BIN)  $(INSTALL_PATH)

clean:
	$(RM) $(OBJ) $(BIN) $(SLIB) *~
