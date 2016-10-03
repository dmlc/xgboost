TEST=test/filesys_test test/dataiter_test\
	test/iostream_test test/recordio_test test/split_read_test\
	test/stream_read_test test/split_test test/libsvm_parser_test\
	 test/split_repeat_read_test test/strtonum_test\
	test/logging_test test/parameter_test test/registry_test\
	test/csv_parser_test

test/filesys_test: test/filesys_test.cc src/io/*.h libdmlc.a
test/dataiter_test: test/dataiter_test.cc  libdmlc.a
test/iostream_test: test/iostream_test.cc libdmlc.a
test/recordio_test: test/recordio_test.cc libdmlc.a
test/split_read_test: test/split_read_test.cc libdmlc.a
test/split_repeat_read_test: test/split_repeat_read_test.cc libdmlc.a
test/stream_read_test: test/stream_read_test.cc libdmlc.a
test/split_test: test/split_test.cc libdmlc.a
test/libsvm_parser_test: test/libsvm_parser_test.cc src/data/libsvm_parser.h libdmlc.a
test/csv_parser_test: test/csv_parser_test.cc src/data/csv_parser.h libdmlc.a
test/strtonum_test: test/strtonum_test.cc src/data/strtonum.h
test/logging_test: test/logging_test.cc
test/parameter_test: test/parameter_test.cc
test/registry_test: test/registry_test.cc

$(TEST) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a,  $^) $(LDFLAGS)

include test/unittest/dmlc_unittest.mk

ALL_TEST=$(TEST) $(UNITTEST)
ALL_TEST_OBJ=$(UNITTEST_OBJ)
