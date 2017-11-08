#include "stb_image_write.h"
#include <cutil_math.h>
#include <curand.h>
#include <curand_kernel.h>

// Dimensions
#define SCREEN_W 1024
#define SCREEN_H 1024
#define NUM_CHANNELS 3

// Renderer constants
#define MAX_BOUNCES 10
#define PUSH_RAY_ORIGIN 0.05f

struct Sphere {
  float3 pos;
  float radius;
  
  // Material
  float3 color;
  float3 emission;
  
};

struct Scene {
  int numObjects;
  Sphere* objects;
};

struct HitData {
  float t;
  // index of object hit
  float index;
};

struct Ray {
  float3 origin;
  float3 direction;
};

__device__ bool intersectSphere(Ray* ray, Sphere* sphere, float* t) {
	float3 offset = ray.origin - sphere.pos;
	float a = dot(ray.direction, ray.direction);
	float b = 2.0 * dot(ray.direction, offset);
	float c = dot(offset, offset) - sphere.radius * sphere.radius;
	float determinant = b*b - 4 * a*c;
	// hit one or both sides
	if (determinant >= 0) {
		float tNear = (-b - sqrt((b*b) - 4.0*a*c))/(2.0*a);
		float tFar = (-b + sqrt((b*b) - 4.0*a*c))/(2.0*a);
		if(tNear > 0 && tFar > 0)
			*t = min(tNear, tFar);
		else if(tNear > 0)
			*t = tNear;
		else
			*t = tFar;
		return true;
	}
	return false;
}

__device__ bool intersectScene(Scene* scene, Ray* ray, Hitdata* hitData) {
  float tNearest = 1000000.0f;
	float t = 0;
	bool hit = false;
	for (int i = 0; i < scene->numObjects; i++) {
		//if there was an intersection and it is the closest
		if (intersectSphere(ray, &scene->objects[i], t) && t > 0 && t < tNearest) {
			tNearest = t;
			hit = true;
			hitData->t = t;
			hitData->index = i;
		}
	}
	return hit;
}

__device__ float3 getCosineWeightedNormal(float3 dir, curandState_t* randState) {
  float power = 1.0f; //0 for unbiased
  dir = norm3df(dir);
	float3 o1 = norm3f(ortho(dir));
	float3 o2 = normalize(cross(dir, o1));
	float2 r = curand2(randState);
	r.x = r.x * 2.0f * PI;
	r.y = pow(r.y, 1.0f / (power + 1.0f));
	float oneminus = sqrt(1.0 - r.y * r.y);
	return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}

__device__ void trace_ray(Scene* scene, Ray* ray, curandState_t* randState) {
  HitData hitData;
  float3 color = float3(0,0,0);
  float3 mask = float3(1,1,1);
  
  for (int n = 0; n < MAX_BOUNCES; n++) {
    // ray leaves the scene
    if (!intersectScene(ray, &hitData))
      return color;
    
    // accumulate color
    color += mask * scene->objects[hitData.index].emission;
    // bounce off surface
    float3 pos = ray->origin + ray->direction * hitData.t;
    float3 normal = norm3df(pos - scene->objects[hitData.index].pos);
    // flip normal if necessary
    normal = dot(normal, ray.direction) < 0 ? normal : -1 * normal;
    // create next ray
    ray->origin = pos + normal * PUSH_RAY_ORIGIN;
    ray->direction = norm3df(getCosineWeightedNormal(normal, randState));
    
    // attenuate color
    mask *= scene->objects[hitData.index].color;
  }
  return color;
}

__global__ void pixel_kernel(float* output, int w, int h, int spp, Scene* scene, curandState_t* randStates) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int id = x * SCREEN_W + y;
  
  if (x >= w || y >= h)
    return;
  
  //copy random state to local memory
  curandState_t localRandState = randStates[id];
  
  // trace ray for each sample
  float3 color;
  color.x = x/(float)SCREEN_W;
  color.y = y/(float)SCREEN_H;
  color.z = 1.0f;
  for (int i = 0; i < spp; i++) {
    // Create ray
    Ray ray;
    color += trace_ray(scene, &ray, &localRandState);
  }
  color /= (float)spp;
  
  // write to output buffer
  output[x*SCREEN_W*NUM_CHANNELS + y*NUM_CHANNELS + 0] = color.x;
  output[x*SCREEN_W*NUM_CHANNELS + y*NUM_CHANNELS + 1] = color.y;
  output[x*SCREEN_W*NUM_CHANNELS + y*NUM_CHANNELS + 2] = color.z;
  // copy rand state back to global memory
  randStates[id] = localRandState;
}

__global__ void setup_random(curandState_t* states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int id = x * SCREEN_W + y;
  curand_init(0, id, 0, &state[id]);
}


int main(int argc, char** argv) {
  // allocate output buffer on host and device
  float* screenBuffer = new float[SCREEN_W*SCREEN_H*NUM_CHANNELS];
  float* d_screenBuffer;
  cudaMalloc(&d_screenBuffer, SCREEN_W*SCREEN_H*NUM_CHANNELS*sizeof(float));
  
  // determine how to distribute work to GPU
  int bx = (SCREEN_W + blockSize.x - 1)/blockSize.x;
  int by = (SCREEN_H + blockSize.y – 1)/blockSize.y;
  dim3 gridSize = dim3(bx, by);
  
  // random number generator states: 1 for each pixel/thread
  curandState_t* d_states;
  int numCurandStates = (bx * blockSize.x) * (by * blockSize.y);
  cudaMalloc(&d_states, numCurandStates * sizeof(curandState_t));
  setup_random<<<gridSize, blockSize>>>(d_states);
  
  // create scene
  Scene scene;
  scene.numObjects = 0;
  scene.objects = new 
  
  // run kernel
  pixel_kernel<<<gridSize, blockSize>>>(screenBuffer, SCREEN_W, SCREEN_H, 1, d_scene);
  
  // copy output buffer back to host
  cudaMemcpy(screenBuffer, d_screenBuffer, SCREEN_W*SCREEN_H*NUM_CHANNELS*sizeof(float), cudaMemcpyDeviceToHost);
  
  // save bitmap
  unsigned char* outBuffer = new unsigned char[SCREEN_W*SCREEN_H*3];
  for (int i = 0; i < SCREEN_W*SCREEN_H*3; i++)
    outBuffer = (unsigned char)fminf(0.0f, fmaxf(255.0f, (255.0f * screenBuffer)));
  stbi_write_bmp("output.bmp", SCREEN_W, SCREEN_H, 3, outBuffer);
  
  // clean up
  cudaFree(d_screenBuffer);
  delete[] d_screenBuffer;
  return 0;
}