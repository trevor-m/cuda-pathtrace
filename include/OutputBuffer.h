#ifndef OUTPUTBUFFER_H
#define OUTPUTBUFFER_H

#include "CudaErrorCheck.h"
#include "tinyexr.h"
#include "stb_image_write.h"
#include <vector>
#include <string>

// Full pixel buffer for all channels
class OutputBuffer {
private:
  void saveFeatureToBitmap(std::string filename, int feature, int channels) {
    unsigned char* outBuffer = new unsigned char[width*height*channels];
    for (int x = 0; x < width; x++)
      for (int y = 0; y < height; y++)
        for (int c = 0; c < channels; c++)
          outBuffer[x*width*channels + y*channels + c] = (unsigned char)min(255, max(0, (int)(255.0f * buffer[x*width*14 + y*14 + feature + c])));

    stbi_write_bmp(filename.c_str(), width, height, channels, outBuffer);
    delete[] outBuffer;
  }

public:
  int width, height;
  float* buffer; // 14 channels
  /*float* color; // 3 channels
  float* normal; // 3 channels
  float* albedo; // 3 channels
  float* depth; // 1 channel
  float* color_var; // 1 channel (Luminance)
  float* normal_var; // 1 channel (Luminance)
  float* albedo_var; // 1 channel (Luminance)
  float* depth_var; // 1 channel*/

  OutputBuffer() {
    width = height = -1;
    buffer = NULL;
    //color = normal = albedo = depth = color_var = normal_var = albedo_var = depth_var = NULL;
  }

  OutputBuffer(int width, int height) {
    this->width = width;
    this->height = height;
    buffer = NULL;
    //color = normal = albedo = depth = color_var = normal_var = albedo_var = depth_var = NULL;
  }

  void CopyFromGPU(const OutputBuffer& d_buffer) {
    gpuErrchk(cudaMemcpy(buffer, d_buffer.buffer, width*height*14*sizeof(float), cudaMemcpyDeviceToHost));
    /*gpuErrchk(cudaMemcpy(color, d_buffer.color, width*height*3*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(normal, d_buffer.normal, width*height*3*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(albedo, d_buffer.albedo, width*height*3*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(depth, d_buffer.depth, width*height*1*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(color_var, d_buffer.color_var, width*height*1*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(normal_var, d_buffer.normal_var, width*height*1*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(albedo_var, d_buffer.albedo_var, width*height*1*sizeof(float), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(depth_var, d_buffer.depth_var, width*height*1*sizeof(float), cudaMemcpyDeviceToHost));*/
  }

  void AllocateCPU() {
    buffer = new float[width*height*14];
    /*color = new float[width*height*3];
    normal = new float[width*height*3];
    albedo = new float[width*height*3];
    depth = new float[width*height*1];
    color_var = new float[width*height*1];
    normal_var = new float[width*height*1];
    albedo_var = new float[width*height*1];
    depth_var = new float[width*height*1];*/
  }

  void AllocateGPU() {
    gpuErrchk(cudaMalloc(&buffer, width*height*14*sizeof(float)));
    /*gpuErrchk(cudaMalloc(&color, width*height*3*sizeof(float)));
    gpuErrchk(cudaMalloc(&normal, width*height*3*sizeof(float)));
    gpuErrchk(cudaMalloc(&albedo, width*height*3*sizeof(float)));
    gpuErrchk(cudaMalloc(&depth, width*height*1*sizeof(float)));
    gpuErrchk(cudaMalloc(&color_var, width*height*1*sizeof(float)));
    gpuErrchk(cudaMalloc(&normal_var, width*height*1*sizeof(float)));
    gpuErrchk(cudaMalloc(&albedo_var, width*height*1*sizeof(float)));
    gpuErrchk(cudaMalloc(&depth_var, width*height*1*sizeof(float)));*/
  }

  void SaveBitmaps(std::string filenameBase) {
    saveFeatureToBitmap(filenameBase+"_color.bmp", 0, 3);
    saveFeatureToBitmap(filenameBase+"_normal.bmp", 3, 3);
    saveFeatureToBitmap(filenameBase+"_albedo.bmp", 6, 3);
    saveFeatureToBitmap(filenameBase+"_depth.bmp", 9, 1);
    saveFeatureToBitmap(filenameBase+"_color_var.bmp", 10, 1);
    saveFeatureToBitmap(filenameBase+"_normal_var.bmp", 11, 1);
    saveFeatureToBitmap(filenameBase+"_albedo_var.bmp", 12, 1);
    saveFeatureToBitmap(filenameBase+"_depth_var.bmp", 13, 1);
  }

  void FreeCPU() {
    delete[] buffer;
    /*delete[] color;
    delete[] normal;
    delete[] albedo;
    delete[] depth;
    delete[] color_var;
    delete[] normal_var;
    delete[] albedo_var;
    delete[] depth_var;*/
  }

  void FreeGPU() {
    cudaFree(buffer);
    /*cudaFree(color);
    cudaFree(normal);
    cudaFree(albedo);
    cudaFree(depth);
    cudaFree(color_var);
    cudaFree(normal_var);
    cudaFree(albedo_var);
    cudaFree(depth_var);*/
  }

  void SaveEXR(std::string filename) {
    EXRHeader header;
    InitEXRHeader(&header);

    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 14;

    // Temporary buffers to split RGB features
    std::vector<float> images[14];
    for(int i = 0; i < 14; i++)
      images[i].resize(width * height);
    // Split RGBRGBRGB... into R, G and B layer for RGB features
    int i = 0;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        for (int c = 0; c < 14; c++)
          images[c][i] = buffer[x*width*14 + y*14 + c];
        i++;
      }
    }

    // determine arrangment of channels
    // note: for some reason, channels must be in alphabetical order
    float* image_ptr[14];
    image_ptr[0] = &(images[8].at(0)); // Albedo.B
    image_ptr[1] = &(images[7].at(0)); // Albedo.G
    image_ptr[2] = &(images[6].at(0)); // Albedo.R
    image_ptr[3] = &(images[12].at(0)); //albedo var
    image_ptr[4] = &(images[2].at(0)); // Color.B
    image_ptr[5] = &(images[1].at(0)); // Color.G
    image_ptr[6] = &(images[0].at(0)); // Color.R
    image_ptr[7] = &(images[10].at(0)); // color var
    image_ptr[8] = &(images[9].at(0)); //depth
    image_ptr[9] = &(images[13].at(0)); //depth var
    image_ptr[10] = &(images[5].at(0)); // Normal.Z
    image_ptr[11] = &(images[4].at(0)); // Normal.Y
    image_ptr[12] = &(images[3].at(0)); // Normal.X
    image_ptr[13] = &(images[11].at(0)); //normal var

    image.images = (unsigned char**)image_ptr;
    image.width = width;
    image.height = height;

    header.num_channels = 14;
    header.channels = (EXRChannelInfo *)malloc(sizeof(EXRChannelInfo) * header.num_channels); 
    // Must be (A)BGR order, since most of EXR viewers expect this channel order.
    strncpy(header.channels[0].name, "Albedo.B", 255); header.channels[0].name[strlen("Albedo.B")] = '\0';
    strncpy(header.channels[1].name, "Albedo.G", 255); header.channels[1].name[strlen("Albedo.G")] = '\0';
    strncpy(header.channels[2].name, "Albedo.R", 255); header.channels[2].name[strlen("Albedo.R")] = '\0';
    strncpy(header.channels[3].name, "AlbedoVar.Z", 255); header.channels[3].name[strlen("AlbedoVar.Y")] = '\0';
    strncpy(header.channels[4].name, "Color.B", 255); header.channels[4].name[strlen("Color.B")] = '\0';
    strncpy(header.channels[5].name, "Color.G", 255); header.channels[5].name[strlen("Color.G")] = '\0';
    strncpy(header.channels[6].name, "Color.R", 255); header.channels[6].name[strlen("Color.R")] = '\0';
    strncpy(header.channels[7].name, "ColorVar.Z", 255); header.channels[7].name[strlen("ColorVar.Y")] = '\0';
    strncpy(header.channels[8].name, "Depth.Z", 255); header.channels[8].name[strlen("Depth.Z")] = '\0';
    strncpy(header.channels[9].name, "DepthVar.Z", 255); header.channels[9].name[strlen("DepthVar.Z")] = '\0';
    strncpy(header.channels[10].name, "Normal.Z", 255); header.channels[10].name[strlen("Normal.Z")] = '\0';
    strncpy(header.channels[11].name, "Normal.Y", 255); header.channels[11].name[strlen("Normal.Y")] = '\0';
    strncpy(header.channels[12].name, "Normal.X", 255); header.channels[12].name[strlen("Normal.X")] = '\0';
    strncpy(header.channels[13].name, "NormalVar.Z", 255); header.channels[13].name[strlen("NormalVar.Y")] = '\0';

    header.pixel_types = (int *)malloc(sizeof(int) * header.num_channels); 
    header.requested_pixel_types = (int *)malloc(sizeof(int) * header.num_channels);
    for (int i = 0; i < header.num_channels; i++) {
      header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of input image
      header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT; // pixel type of output image to be stored in .EXR
    }

    const char* err;
    int ret = SaveEXRImageToFile(&image, &header, filename.c_str(), &err);
    if (ret != TINYEXR_SUCCESS) {
      fprintf(stderr, "Error saving EXR: %s\n", err);
      return;
    }
    //printf("Saved exr file. [ %s ] \n", filename.c_str());

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);
  }
};

#endif