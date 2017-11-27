#ifndef DENOISER_H
#define DENOISER_H

#include "OutputBuffer.h"
#include "GLPixelBuffer.h"
#include "denoise.h"

class Denoiser {
private:
  int width, height;

  // work distribution
  dim3 gridSize, dimBlock;

public:

  Denoiser(int width, int height, int numThreads) {
    this->width = width;
    this->height = height;

    // determine how to distribute work to GPU
    int blockSize = numThreads;
    int bx = (width + blockSize - 1)/blockSize;
    int by = (height + blockSize - 1)/blockSize;
    gridSize = dim3(bx, by);
    dimBlock = dim3(blockSize, blockSize);
  }

  float Denoise(const OutputBuffer& d_buffer, GLPixelBuffer denoisedBuffer) {
    // create buffer to output to which is mapped to an opengl texture
    cudaThreadSynchronize();
    float* d_output;
    check_gl_error();
    denoisedBuffer.MapToGPU(&d_output);
    
    // launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // run kernel
    cudaEventRecord(start);
    denoise_kernel<<<gridSize, dimBlock>>>(d_buffer, d_output);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //unmap
    cudaThreadSynchronize();
    denoisedBuffer.UnMap();

    return milliseconds;
  }
};

#endif