#include "Camera.h"
#include "pathtrace.h"
#include <iostream>
#include <string>
#include <vector>
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


void saveBufferToBMP(std::string filename, float* data, int channels) {
  unsigned char* outBuffer = new unsigned char[SCREEN_W*SCREEN_H*channels];
  for (int i = 0; i < SCREEN_W*SCREEN_H*channels; i++)
    outBuffer[i] = (unsigned char)min(255, max(0, (int)(255.0f * data[i])));

  stbi_write_bmp(filename.c_str(), SCREEN_W, SCREEN_H, channels, outBuffer);
  delete[] outBuffer;
}

void saveBuffersToEXR(std::string filename, const OutputBuffer& buffer) {
  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 14;
  int width = SCREEN_W;
  int height = SCREEN_H;

  // Temporary buffers to split RGB features
  std::vector<float> images[9];
  for(int i = 0; i < 9; i++)
    images[i].resize(width * height);
  // Split RGBRGBRGB... into R, G and B layer for RGB features
  for (int i = 0; i < width * height; i++) {
    images[0][i] = buffer.color[3*i+0];
    images[1][i] = buffer.color[3*i+1];
    images[2][i] = buffer.color[3*i+2];
    images[3][i] = buffer.normal[3*i+0];
    images[4][i] = buffer.normal[3*i+1];
    images[5][i] = buffer.normal[3*i+2];
    images[6][i] = buffer.albedo[3*i+0];
    images[7][i] = buffer.albedo[3*i+1];
    images[8][i] = buffer.albedo[3*i+2];
  }

  float* image_ptr[14];
  image_ptr[0] = &(images[8].at(0)); // Albedo.B
  image_ptr[1] = &(images[7].at(0)); // Albedo.G
  image_ptr[2] = &(images[6].at(0)); // Albedo.R
  image_ptr[3] = buffer.albedo_var;
  image_ptr[4] = &(images[2].at(0)); // Color.B
  image_ptr[5] = &(images[1].at(0)); // Color.G
  image_ptr[6] = &(images[0].at(0)); // Color.R
  image_ptr[7] = buffer.color_var;
  image_ptr[8] = buffer.depth;
  image_ptr[9] = buffer.depth_var;
  image_ptr[10] = &(images[5].at(0)); // Normal.Z
  image_ptr[11] = &(images[4].at(0)); // Normal.Y
  image_ptr[12] = &(images[3].at(0)); // Normal.X
  image_ptr[13] = buffer.normal_var;


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
  printf("Saved exr file. [ %s ] \n", filename.c_str());

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);
}

int main(int argc, const char** argv) {
  //get arguments
  int samplesPerPixel = SAMPLES;
  int cudaDevice = 0;
  std::string outputName = "output";
  if(argc == 2) {
    samplesPerPixel = std::stoi(argv[1]);
  }
  else if(argc == 3) {
    samplesPerPixel = std::stoi(argv[1]);
    outputName = std::string(argv[2]);
  }
  else if(argc == 4) {
    samplesPerPixel = std::stoi(argv[1]);
    outputName = std::string(argv[2]);
    cudaDevice = std::stoi(argv[3]);
  }
  else {
    std::cout << "Usage:" << std::endl;
    std::cout << "\tpathtrace <samples per pixel> <output name>" << std::endl;
    std::cout << "\tpathtrace <samples per pixel> <output name> <CUDA device>" << std::endl;
  }

  std::cout << "Output file prefix: " << outputName << std::endl;
  std::cout << "Samples per pixel: " << samplesPerPixel << std::endl;
  std::cout << "Using CUDA device: " << cudaDevice << std::endl;

  // determine how to distribute work to GPU
  int blockSize = 32;
  int bx = (SCREEN_W + blockSize - 1)/blockSize;
  int by = (SCREEN_H + blockSize - 1)/blockSize;
  dim3 gridSize = dim3(bx, by);
  dim3 dimBlock = dim3(blockSize, blockSize);

  // set cuda device
  gpuErrchk(cudaSetDevice(cudaDevice));
  
  // random number generator states: 1 for each pixel/thread
  curandState* d_states;
  int numCurandStates = SCREEN_W*SCREEN_H;
  gpuErrchk(cudaMalloc(&d_states, numCurandStates * sizeof(curandState)));
  setup_random<<<gridSize, dimBlock>>>(d_states);
  
  // create scene
  Scene d_scene;
  d_scene.numObjects = 9;
  Sphere spheres[] = {
   { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f } }, //Left 
   { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f } }, //Right 
   { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Back 
   { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f } }, //Frnt 
   { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Botm 
   { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Top 
   { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } }, // small sphere 1
   { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } }, // small sphere 2
   { 600.0f, { 50.0f, 681.6f - .78f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f } }  // Light y=-/77 originally
  };
  gpuErrchk(cudaMalloc(&d_scene.objects, d_scene.numObjects*sizeof(Sphere)));
  gpuErrchk(cudaMemcpy(d_scene.objects, spheres, d_scene.numObjects*sizeof(Sphere), cudaMemcpyHostToDevice));

  // create camera and compute eye ray basis
  Camera camera(glm::vec3(50, 52, 295.6));
  float3 eyeRayBasis[4];
  camera.getEyeRayBasis(eyeRayBasis, SCREEN_W, SCREEN_H);
  float3* d_eyeRayBasis;
  gpuErrchk(cudaMalloc(&d_eyeRayBasis, 4*sizeof(float3)));
  gpuErrchk(cudaMemcpy(d_eyeRayBasis, eyeRayBasis, 4*sizeof(float3), cudaMemcpyHostToDevice));
  float3* d_eyePos;
  gpuErrchk(cudaMalloc(&d_eyePos, sizeof(float3)));
  gpuErrchk(cudaMemcpy(d_eyePos, &camera.Position, sizeof(float3), cudaMemcpyHostToDevice));
  
  // allocate output buffer on device
  OutputBuffer d_buffer;
  gpuErrchk(cudaMalloc(&d_buffer.color, SCREEN_W*SCREEN_H*3*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_buffer.normal, SCREEN_W*SCREEN_H*3*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_buffer.albedo, SCREEN_W*SCREEN_H*3*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_buffer.depth, SCREEN_W*SCREEN_H*1*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_buffer.color_var, SCREEN_W*SCREEN_H*1*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_buffer.normal_var, SCREEN_W*SCREEN_H*1*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_buffer.albedo_var, SCREEN_W*SCREEN_H*1*sizeof(float)));
  gpuErrchk(cudaMalloc(&d_buffer.depth_var, SCREEN_W*SCREEN_H*1*sizeof(float)));

  //measure how long kernel takes
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // run kernel
  cudaEventRecord(start);
  pixel_kernel<<<gridSize, dimBlock>>>(d_buffer, d_states, d_scene, d_eyeRayBasis, d_eyePos, samplesPerPixel);
  cudaEventRecord(stop);

  // copy output buffer back to host
  OutputBuffer buffer;
  buffer.color = new float[SCREEN_W*SCREEN_H*3];
  buffer.normal = new float[SCREEN_W*SCREEN_H*3];
  buffer.albedo = new float[SCREEN_W*SCREEN_H*3];
  buffer.depth = new float[SCREEN_W*SCREEN_H*1];
  buffer.color_var = new float[SCREEN_W*SCREEN_H*1];
  buffer.normal_var = new float[SCREEN_W*SCREEN_H*1];
  buffer.albedo_var = new float[SCREEN_W*SCREEN_H*1];
  buffer.depth_var = new float[SCREEN_W*SCREEN_H*1];
  gpuErrchk(cudaMemcpy(buffer.color, d_buffer.color, SCREEN_W*SCREEN_H*3*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buffer.normal, d_buffer.normal, SCREEN_W*SCREEN_H*3*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buffer.albedo, d_buffer.albedo, SCREEN_W*SCREEN_H*3*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buffer.depth, d_buffer.depth, SCREEN_W*SCREEN_H*1*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buffer.color_var, d_buffer.color_var, SCREEN_W*SCREEN_H*1*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buffer.normal_var, d_buffer.normal_var, SCREEN_W*SCREEN_H*1*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buffer.albedo_var, d_buffer.albedo_var, SCREEN_W*SCREEN_H*1*sizeof(float), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(buffer.depth_var, d_buffer.depth_var, SCREEN_W*SCREEN_H*1*sizeof(float), cudaMemcpyDeviceToHost));

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel took %fms (%f fps)\n", milliseconds, 1000.0f/milliseconds);

  // save EXR
  saveBuffersToEXR(outputName+".exr", buffer);

  // save bitmaps
  saveBufferToBMP(outputName+"_color.bmp", buffer.color, 3);
  saveBufferToBMP(outputName+"_normal.bmp", buffer.normal, 3);
  saveBufferToBMP(outputName+"_albedo.bmp", buffer.albedo, 3);
  saveBufferToBMP(outputName+"_depth.bmp", buffer.depth, 1);
  saveBufferToBMP(outputName+"_color_var.bmp", buffer.color_var, 1);
  saveBufferToBMP(outputName+"_normal_var.bmp", buffer.normal_var, 1);
  saveBufferToBMP(outputName+"_albedo_var.bmp", buffer.albedo_var, 1);
  saveBufferToBMP(outputName+"_depth_var.bmp", buffer.depth_var, 1);
  


  // clean up
  //cudaFree(d_buffer.color);
  //delete[] screenBuffer;
  return 0;
}
