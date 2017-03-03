#include "mex.h"
#include "matrix.h"
#include <lz4.h>

void mexFunction(int nlhs, mxArray *plhs[],
		 int nrhs, const mxArray *prhs[])
{
  double d_sz = mxGetScalar(prhs[0]);
  int sz      = (int)(d_sz+0.5);
  char *src   = (char*)mxGetData(prhs[1]);

  plhs[0]     = mxCreateNumericMatrix(sz, 1, mxUINT8_CLASS, mxREAL);
  char *dst   = (char *)mxGetData(plhs[0]);

  //LZ4_decompress_fast (const char* source, char* dest, int originalSize);
  LZ4_decompress_fast(src, dst, sz);
}
