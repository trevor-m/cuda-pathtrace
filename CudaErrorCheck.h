#ifndef CUDA_ERROR_CHECK_H
#define CUDA_ERROR_CHECK_H
#include <stdio.h>

// Helper function to run cuda function and check for errors
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void _check_gl_error(const char *file, int line);

#define check_gl_error() _check_gl_error(__FILE__,__LINE__)

#endif