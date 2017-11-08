#include "Camera.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <helper_math.h>
#include <math_constants.h>
#include <curand.h>
#include <curand_kernel.h>
#include <iostream>

// Dimensions
#define SCREEN_W 512
#define SCREEN_H 512
#define NUM_CHANNELS 3

// Renderer constants
#define SAMPLES 4096
#define MAX_BOUNCES 10
#define PUSH_RAY_ORIGIN 0.05f

#include <helper_cuda.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct Sphere {
  float radius;
  float3 pos;
  
  // Material
  float3 emission;
  float3 color;
};

struct Scene {
  int numObjects;
  Sphere* objects;
};

struct HitData {
  float t;
  // index of object hit
  int index;
};

struct Ray {
  float3 origin;
  float3 direction;
};

__device__ bool intersectSphere(const Ray& ray, const Sphere& sphere, float* t) {
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

__device__ bool intersectScene(const Scene& scene, const Ray& ray, HitData* hitData) {
  float tNearest = 1000000.0f;
	float t = 0;
	bool hit = false;
	for (int i = 0; i < scene.numObjects; i++) {
		//if there was an intersection and it is the closest
		if (intersectSphere(ray, scene.objects[i], &t) && t > 0 && t < tNearest) {
			tNearest = t;
			hit = true;
			hitData->t = t;
			hitData->index = i;
		}
	}
	return hit;
}

__device__ float3 orthoVector(float3 v) {
    //  See : http://lolengine.net/blog/2013/09/21/picking-orthogonal-vector-combing-coconuts
    return (abs(v.x) > abs(v.z)) ? make_float3(-v.y, v.x, 0.0f)  : make_float3(0.0f, -v.z, v.y);
}

__device__ float3 getCosineWeightedNormal(float3 dir, curandState* randState) {
  float power = 1.0f; //0 for unbiased
  dir = normalize(dir);
	float3 o1 = normalize(orthoVector(dir));
	float3 o2 = normalize(cross(dir, o1));
	float2 r = make_float2(curand_uniform(randState), curand_uniform(randState));
	r.x = r.x * 2.0f * CUDART_PI_F;
	r.y = pow(r.y, 1.0f / (power + 1.0f));
	float oneminus = sqrt(1.0 - r.y * r.y);
	return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}

__device__ float3 trace_ray(const Scene& scene, Ray ray, curandState* randState) {
  HitData hitData;
  float3 color = make_float3(0,0,0);
  float3 mask = make_float3(1,1,1);
  
  for (int n = 0; n < MAX_BOUNCES; n++) {
    // ray leaves the scene
    if (!intersectScene(scene, ray, &hitData))
      return color;
    
    // accumulate emmission
    color += mask * scene.objects[hitData.index].emission;
    // attenuate color for next bounce
    mask *= scene.objects[hitData.index].color; //account for incoming direction??

    // bounce off surface
    float3 pos = ray.origin + ray.direction * hitData.t;
    float3 normal = normalize(pos - scene.objects[hitData.index].pos);
    // flip normal if necessary
    normal = dot(normal, ray.direction) < 0 ? normal : -1 * normal;
    // create next ray
    ray.origin = pos + normal * PUSH_RAY_ORIGIN;
    ray.direction = normalize(getCosineWeightedNormal(normal, randState));
  }
  return color;
}

__global__ void pixel_kernel(float* output, curandState* randStates, Scene scene, float3* rayBasis, float3* eyePos, int spp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int id = x * SCREEN_W + y;

  if (x >= SCREEN_W || y >= SCREEN_H)
    return;
  
  //copy random state to local memory
  curandState localRandState = randStates[id];
  
  // take samples
  float3 color = make_float3(0.0f, 0.0f, 0.0f);
  for (int i = 0; i < spp; i++) {
    // determine ray direction by interpolating from basis
    float2 screenPos = make_float2(x + curand_uniform(&localRandState)*2.0f - 1.0f, y + curand_uniform(&localRandState)*2.0f - 1.0f);
    screenPos /= make_float2(SCREEN_W, SCREEN_H);
    Ray ray;
    ray.origin = *eyePos;
    ray.direction = lerp(lerp(rayBasis[0], rayBasis[1], screenPos.y), lerp(rayBasis[2], rayBasis[3], screenPos.y), 1.0f-screenPos.x);
    // trace ray and accumulate color
    color += trace_ray(scene, ray, &localRandState);
  }
  color /= (float)spp;
  
  // write to output buffer
  output[x*SCREEN_W*NUM_CHANNELS + y*NUM_CHANNELS + 0] = color.x;
  output[x*SCREEN_W*NUM_CHANNELS + y*NUM_CHANNELS + 1] = color.y;
  output[x*SCREEN_W*NUM_CHANNELS + y*NUM_CHANNELS + 2] = color.z;
  // copy rand state back to global memory
  randStates[id] = localRandState;
}

__global__ void setup_random(curandState* states) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int id = x * SCREEN_W + y;
  if (x >= SCREEN_W || y >= SCREEN_H)
    return;
  curand_init(id, 0, 0, &states[id]);
}


int main(int argc, char** argv) {
  // determine how to distribute work to GPU
  int blockSize = 32;
  int bx = (SCREEN_W + blockSize - 1)/blockSize;
  int by = (SCREEN_H + blockSize - 1)/blockSize;
  dim3 gridSize = dim3(bx, by);
  dim3 dimBlock = dim3(blockSize, blockSize);

  std::cout << gridSize.x << ", " << gridSize.y << std::endl;

  gpuErrchk(cudaSetDevice(1));
  
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
   { 600.0f, { 50.0f, 681.6f - .77f, 81.6f }, { 2.0f, 1.8f, 1.6f }, { 0.0f, 0.0f, 0.0f } }  // Light
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
  
  // allocate output buffer on host and device
  float* screenBuffer = new float[SCREEN_W*SCREEN_H*NUM_CHANNELS];
  float* d_screenBuffer;
  gpuErrchk(cudaMalloc(&d_screenBuffer, SCREEN_W*SCREEN_H*NUM_CHANNELS*sizeof(float)));

  //measure how long kernel takes
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // run kernel
  cudaEventRecord(start);
  pixel_kernel<<<gridSize, dimBlock>>>(d_screenBuffer, d_states, d_scene, d_eyeRayBasis, d_eyePos, SAMPLES);
  cudaEventRecord(stop);

  // copy output buffer back to host
  gpuErrchk(cudaMemcpy(screenBuffer, d_screenBuffer, SCREEN_W*SCREEN_H*NUM_CHANNELS*sizeof(float), cudaMemcpyDeviceToHost));

  // save bitmap
  unsigned char* outBuffer = new unsigned char[SCREEN_W*SCREEN_H*3];
  for (int i = 0; i < SCREEN_W*SCREEN_H*3; i++)
    outBuffer[i] = (unsigned char)min(255, max(0, (int)(255.0f * screenBuffer[i])));

  stbi_write_bmp("output.bmp", SCREEN_W, SCREEN_H, 3, outBuffer);
  delete[] outBuffer;
  
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel took %fms (%f fps)\n", milliseconds, 1000.0f/milliseconds);

  // clean up
  cudaFree(d_screenBuffer);
  delete[] screenBuffer;
  return 0;
}
