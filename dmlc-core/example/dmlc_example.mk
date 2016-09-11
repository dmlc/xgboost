ALL_EXAMPLE=example/parameter


example/parameter: example/parameter.cc libdmlc.a

$(ALL_EXAMPLE) :
	$(CXX) $(CFLAGS) -o $@ $(filter %.cpp %.o %.c %.cc %.a,  $^) $(LDFLAGS)


