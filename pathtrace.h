#ifndef PATHTRACE_H
#define PATHTRACE_H
#include <helper_cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Default dimensions
#define SCREEN_W 512
#define SCREEN_H 512
#define SAMPLES 4

// A sphere object
struct Sphere {
  float radius;
  float3 pos;
  
  // Material
  float3 emission;
  float3 color;
};

// A scene that can be rendered
struct Scene {
  int numObjects;
  Sphere* objects;
};

// Full pixel buffer of all channels
struct OutputBuffer {
  float* color; // 3 channels
  float* normal; // 3 channels
  float* albedo; // 3 channels
  float* depth; // 1 channel
  float* color_var; // 1 channel (Luminance)
  float* normal_var; // 1 channel (Luminance)
  float* albedo_var; // 1 channel (Luminance)
  float* depth_var; // 1 channel
};


// Kernel to compute the color of a pixel
__global__ void pixel_kernel(OutputBuffer output, curandState* randStates, Scene scene, float3* rayBasis, float3* eyePos, int spp);

// Intialize random generator states
__global__ void setup_random(curandState* states);


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

#endif