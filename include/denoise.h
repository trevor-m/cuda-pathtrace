#ifndef DENOISE_H
#define DENOISE_H
#include "OutputBuffer.h"

// Kernel to compute the color of a pixel
__global__ void denoise_kernel(OutputBuffer input, float* d_output);

#endif