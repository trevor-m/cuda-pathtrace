#ifndef PATHTRACE_H
#define PATHTRACE_H
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "Scene.h"
#include "OutputBuffer.h"

// Kernel to compute the color of a pixel
__global__ void pixel_kernel(OutputBuffer output, curandState* randStates, Scene scene, float3* rayBasis, float3* eyePos, int spp);

// Intialize random generator states
__global__ void setup_random(curandState* states, int width, int height);

#endif