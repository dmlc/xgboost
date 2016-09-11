UTEST_ROOT=test/unittest
UNITTEST=$(UTEST_ROOT)/dmlc_unittest
UNITTEST_SRC=$(wildcard $(UTEST_ROOT)/*.cc)
UNITTEST_OBJ=$(patsubst %.cc,%.o,$(UNITTEST_SRC))

GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

$(UTEST_ROOT)/%.o : $(UTEST_ROOT)/%.cc libdmlc.a
	$(CXX) $(CFLAGS) -I$(GTEST_INC) -o $@ -c $<

$(UNITTEST) : $(UNITTEST_OBJ)
	$(CXX) $(CFLAGS) -L$(GTEST_LIB) -o $@ $^ libdmlc.a $(LDFLAGS) -lgtest
