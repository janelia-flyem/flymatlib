MEX = /misc/local/matlab-2015b/bin/mex

all: fml_lz4_mex.mexa64

fml_lz4_mex.mexa64: fml_lz4_mex.cpp
	@CFLAGS="-fPIC" $(MAKE) -C lz4/lib liblz4.a
	$(MEX) -largeArrayDims -v CXXOPTIMFLAGS='-O3 -DNDEBUG' LDOPTIMFLAGS='-O3' -Ilz4/lib lz4/lib/liblz4.a fml_lz4_mex.cpp

clean:
	rm fml_lz4_mex.mexa64
