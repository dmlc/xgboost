UTEST_ROOT=tests/cpp
UTEST_OBJ_ROOT=build_$(UTEST_ROOT)
UNITTEST=$(UTEST_ROOT)/xgboost_test

UNITTEST_SRC=$(wildcard $(UTEST_ROOT)/*.cc $(UTEST_ROOT)/*/*.cc)
UNITTEST_OBJ=$(patsubst $(UTEST_ROOT)%.cc, $(UTEST_OBJ_ROOT)%.o, $(UNITTEST_SRC))

GTEST_LIB=$(GTEST_PATH)/lib/
GTEST_INC=$(GTEST_PATH)/include/

UNITTEST_CFLAGS=$(CFLAGS)
UNITTEST_LDFLAGS=$(LDFLAGS) -L$(GTEST_LIB) -lgtest
UNITTEST_DEPS=lib/libxgboost.a $(DMLC_CORE)/libdmlc.a $(RABIT)/lib/$(LIB_RABIT)

COVER_OBJ=$(patsubst %.o, %.gcda, $(ALL_OBJ)) $(patsubst %.o, %.gcda, $(UNITTEST_OBJ))

$(UTEST_OBJ_ROOT)/$(GTEST_PATH)/%.o: $(GTEST_PATH)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(UNITTEST_CFLAGS) -I$(GTEST_INC) -I$(GTEST_PATH) -o $@ -c $<

$(UTEST_OBJ_ROOT)/%.o: $(UTEST_ROOT)/%.cc
	@mkdir -p $(@D)
	$(CXX) $(UNITTEST_CFLAGS) -I$(GTEST_INC) -o $@ -c $<

$(UNITTEST): $(UNITTEST_OBJ) $(UNITTEST_DEPS)
	$(CXX) $(UNITTEST_CFLAGS) -o $@ $^ $(UNITTEST_LDFLAGS)


ALL_TEST=$(UNITTEST)
ALL_TEST_OBJ=$(UNITTEST_OBJ)
