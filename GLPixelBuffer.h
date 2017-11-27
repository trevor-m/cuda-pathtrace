#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"
#include "CudaErrorCheck.h"

class GLPixelBuffer {
private:
  int width, height;
	GLuint bufferID, textureID;
	
public:
	GLPixelBuffer(int width, int height) {
		this->width = width;
		this->height = height;
		// create buffer
		glGenBuffers(1, &bufferID);
		glBindBuffer(GL_ARRAY_BUFFER, bufferID);
		glBufferData(GL_ARRAY_BUFFER, width * height * 3*sizeof(float), NULL, GL_DYNAMIC_DRAW);
		// register with CUDA
		gpuErrchk(cudaGLRegisterBufferObject(bufferID));
		check_gl_error();
	}

	~GLPixelBuffer() {
		//cudaGLUnregisterBufferObject(bufferID);
		//glDeleteBuffers(1, &bufferID);
		glDeleteTextures(1, &textureID);
	}

	void MapToGPU(float** d_buffer) {
		//gpuErrchk(cudaGLMapBufferObject((void**)d_buffer, bufferID));
		gpuErrchk(cudaGLMapBufferObject((void**)d_buffer, bufferID));

		check_gl_error();
	}

	void UnMap() {
	  gpuErrchk(cudaGLUnmapBufferObject(bufferID));
		check_gl_error();
	}

	void BindBuffer() {
		glBindBuffer(GL_ARRAY_BUFFER, bufferID); 
		check_gl_error();
	}
};