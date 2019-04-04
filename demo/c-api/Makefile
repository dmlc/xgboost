SRC=c-api-demo.c
TGT=c-api-demo

cc=cc
CFLAGS ?=-O3
XGBOOST_ROOT ?=../..
INCLUDE_DIR=-I$(XGBOOST_ROOT)/include -I$(XGBOOST_ROOT)/dmlc-core/include -I$(XGBOOST_ROOT)/rabit/include
LIB_DIR=-L$(XGBOOST_ROOT)/lib

build: $(TGT)

$(TGT): $(SRC) Makefile
	$(cc) $(CFLAGS) $(INCLUDE_DIR) $(LIB_DIR) -o $(TGT) $(SRC) -lxgboost

run: $(TGT)
	LD_LIBRARY_PATH=$(XGBOOST_ROOT)/lib ./$(TGT)

clean:
	rm -f $(TGT)
