#include "denoise.h"

union Color  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

__global__ void denoise_kernel(OutputBuffer input, float* d_output) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= input.width || y >= input.height)
    return;

  float3 color = make_float3(input.buffer[x*input.width*14 + y*14 + 0], input.buffer[x*input.width*14 + y*14 + 1], input.buffer[x*input.width*14 + y*14 + 2]);
  // clamp to range 0-1
  color.x = min(max(color.x, 0.0f), 1.0f);
  color.y = min(max(color.y, 0.0f), 1.0f);
  color.z = min(max(color.z, 0.0f), 1.0f);

  Color formatColor;
  formatColor.components = make_uchar4((unsigned char)(color.x*255.0), (unsigned char)(color.y*255.0), (unsigned char)(color.z*255.0), 1);

  // x and y are mixed up because I modified the eye rays so that the image output would be correct...
  d_output[x*input.width*3 + y*3 + 0] = y;
  d_output[x*input.width*3 + y*3 + 1] = input.width - x;
  d_output[x*input.width*3 + y*3 + 2] = formatColor.c;
}