# Makfile for easily install dependencies

# List of packages here
.PHONY: gtest lz4

# rules for gtest
${CACHE_PREFIX}/include/gtest:
	rm -rf gtest release-1.7.0.zip
	wget https://github.com/google/googletest/archive/release-1.7.0.zip
	unzip release-1.7.0.zip
	mv googletest-release-1.7.0 gtest
	cd gtest; g++ -Iinclude -pthread -c src/gtest-all.cc -o gtest-all.o; cd ..
	ar -rv libgtest.a gtest/gtest-all.o
	mkdir -p ${CACHE_PREFIX}/include ${CACHE_PREFIX}/lib
	cp -r gtest/include/gtest ${CACHE_PREFIX}/include
	mv libgtest.a ${CACHE_PREFIX}/lib
	rm -rf release-1.7.0.zip

gtest: | ${CACHE_PREFIX}/include/gtest

lz4:  ${CACHE_PREFIX}/include/lz4.h

${CACHE_PREFIX}/include/lz4.h:
	rm -rf lz4
	git clone https://github.com/Cyan4973/lz4
	cd lz4; make; make install PREFIX=${CACHE_PREFIX}; cd -
