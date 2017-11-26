#ifndef SCENE_H
#define SCENE_H

#include "CudaErrorCheck.h"

// A sphere object
struct Sphere {
  float radius;
  float3 pos;
  
  // Material
  float3 emission;
  float3 color;
};

// A scene which can be rendered
class Scene {
public:
  int numObjects;
  Sphere* objects;

  Scene() {
    numObjects = 9;
    Sphere spheres[] = {
      { 1e5f, { 1e5f + 1.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { 0.75f, 0.25f, 0.25f } }, //Left 
      { 1e5f, { -1e5f + 99.0f, 40.8f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .25f, .25f, .75f } }, //Right 
      { 1e5f, { 50.0f, 40.8f, 1e5f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Back 
      { 1e5f, { 50.0f, 40.8f, -1e5f + 600.0f }, { 0.0f, 0.0f, 0.0f }, { 1.00f, 1.00f, 1.00f } }, //Frnt 
      { 1e5f, { 50.0f, 1e5f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Botm 
      { 1e5f, { 50.0f, -1e5f + 81.6f, 81.6f }, { 0.0f, 0.0f, 0.0f }, { .75f, .75f, .75f } }, //Top 
      { 16.5f, { 27.0f, 16.5f, 47.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } }, // small sphere 1
      { 16.5f, { 73.0f, 16.5f, 78.0f }, { 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f } }, // small sphere 2
      { 600.0f, { 50.0f, 681.6f - .78f, 81.6f }, { 4.0f, 4.0f, 4.0f }, { 0.0f, 0.0f, 0.0f } }  // Light y=-/77 originally
    };
    gpuErrchk(cudaMalloc(&objects, numObjects*sizeof(Sphere)));
    gpuErrchk(cudaMemcpy(objects, spheres, numObjects*sizeof(Sphere), cudaMemcpyHostToDevice));
  }
};

#endif