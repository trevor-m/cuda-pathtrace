#ifndef RENDERER_H
#define RENDERER_H

#include "CudaErrorCheck.h"
#include "Camera.h"
#include "pathtrace.h"

class Renderer {
private:
  int width, height, samplesPerPixel;

  // work distribution
  dim3 gridSize, dimBlock;

  // Device
  // RNG
  curandState* d_states;
  // Camera eye rays
  float3* d_eyeRayBasis;
  float3* d_eyePos;

public:
  // feature buffers
  OutputBuffer d_buffer;

  Renderer(int width, int height, int samplesPerPixel, int numThreads) {
    this->width = width;
    this->height = height;
    this->samplesPerPixel = samplesPerPixel;

    // determine how to distribute work to GPU
    int blockSize = numThreads;
    int bx = (width + blockSize - 1)/blockSize;
    int by = (height + blockSize - 1)/blockSize;
    gridSize = dim3(bx, by);
    dimBlock = dim3(blockSize, blockSize);
    
    // allocate stuff on device
    // random number generator states: 1 for each pixel/thread
    gpuErrchk(cudaMalloc(&d_states, width*height* sizeof(curandState)));
    setup_random<<<gridSize, dimBlock>>>(d_states, width, height);
    // camera information
    gpuErrchk(cudaMalloc(&d_eyeRayBasis, 4*sizeof(float3)));
    gpuErrchk(cudaMalloc(&d_eyePos, sizeof(float3)));
    // buffers
    d_buffer.width = width;
    d_buffer.height = height;
    d_buffer.AllocateGPU();
  }

  ~Renderer() {
    d_buffer.FreeGPU();
    cudaFree(d_states);
    cudaFree(d_eyeRayBasis);
    cudaFree(d_eyePos);
  }

  float Render(const Scene& d_scene, const Camera& camera) {
    // copy camera information to device
    float3 eyeRayBasis[4];
    camera.getEyeRayBasis(eyeRayBasis, width, height);
    gpuErrchk(cudaMemcpy(d_eyeRayBasis, eyeRayBasis, 4*sizeof(float3), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_eyePos, &camera.Position, sizeof(float3), cudaMemcpyHostToDevice));

    // launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // run kernel
    cudaEventRecord(start);
    pixel_kernel<<<gridSize, dimBlock>>>(d_buffer, d_states, d_scene, d_eyeRayBasis, d_eyePos, samplesPerPixel);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
  }

  void Denoise();

  /*void CopyBufferToScreen() {

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glActiveTexture(GL_TEXTURE0 + RENDER_TEXTURE);
    glBindTexture(GL_TEXTURE_2D, result_texture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, image_width, image_height, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    glActiveTexture(GL_TEXTURE0 + UNUSED_TEXTURE);
    // draw quad
  }*/
};

#endif