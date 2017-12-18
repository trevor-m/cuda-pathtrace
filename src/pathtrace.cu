#include "pathtrace.h"
#include <helper_math.h>
#include <math_constants.h>


// Renderer constants
#define MAX_BOUNCES 5
#define PUSH_RAY_ORIGIN 0.05f

struct HitData {
  float t;
  // index of object hit
  int index;
};

struct Ray {
  float3 origin;
  float3 direction;
};

enum Features {COLOR=0, NORMAL=1, ALBEDO=2, DEPTH=3, NUM_FEATURES=4};

// Output of a single ray trace
struct TraceOutput {
  float3 color;
  float3 normal;
  float3 albedo;
  float depth;

  __device__ TraceOutput() {
    color = make_float3(0.0f, 0.0f, 0.0f);
    normal = make_float3(0.0f, 0.0f, 0.0f);
    albedo = make_float3(0.0f, 0.0f, 0.0f);
    depth = 0.0f;
  }
};

// Compute variance for features
struct OnlineVarianceBuffer {
  int n[NUM_FEATURES];
  float mean[NUM_FEATURES];
  float M2[NUM_FEATURES];

  __device__ OnlineVarianceBuffer() {
    for(int i = 0; i < NUM_FEATURES; i++) {
      n[i] = 0;
      mean[i] = 0.0f;
      M2[i] = 0.0f;
    }
  }

  __device__ void updateVariance(float x, Features feature) {
    n[feature] += 1;
    float delta = x - mean[feature];
    mean[feature] += delta/n[feature];
    float delta2 = x - mean[feature];
    M2[feature] += delta*delta2;
  }

  __device__ float getVariance(Features feature) {
    if(n[feature] < 2)
      return 0.0f;
    return (M2[feature] / (n[feature]-1));
  }
};

__device__ float luminance(float3 color) {
  return 0.2126*color.x + 0.7152*color.y + 0.0722*color.z;
}


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

__device__ float testShadowRay(const Scene& scene, const Ray& ray, int lightIndex) {
  float tNearest = length(scene.objects[lightIndex].pos- make_float3(0, scene.objects[lightIndex].radius, 0) - ray.origin);
	float t = 0;
	for (int i = 0; i < scene.numObjects; i++) {
		//if there was an intersection and it is the closest
		if (i != lightIndex && intersectSphere(ray, scene.objects[i], &t) && t > 0 && t < tNearest) {
			return 0.0f;
		}
	}
	return 1.0f;
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

__device__ float3 getDirectLighting(const Scene& scene, const float3& normal, const float3& position) {
  // TODO: sample random point on light
  // TODO: make lights array
  int lightIndex = 8;
  float3 lightDir = normalize(scene.objects[lightIndex].pos - make_float3(0, scene.objects[lightIndex].radius, 0) - position);
  float diffuse = clamp(dot(lightDir, normal), 0.0, 1.0);
  Ray shadowRay;
  shadowRay.origin = position + normal * PUSH_RAY_ORIGIN;
  shadowRay.direction = lightDir;
  return diffuse * scene.objects[lightIndex].emission * testShadowRay(scene, shadowRay, lightIndex);
}

__device__ void trace_ray(TraceOutput& L, const Scene& scene, Ray ray, curandState* randState, int x, int y, OnlineVarianceBuffer& var) {
  HitData hitData;
  float3 color = make_float3(0,0,0);
  float3 mask = make_float3(1,1,1);
  
  for (int n = 0; n < MAX_BOUNCES; n++) {
    // intersect with surface
    if (!intersectScene(scene, ray, &hitData)) {
      // ray leaves the scene
      L.color += color;
      return;
    }
    // get intersection position and normal
    float3 pos = ray.origin + ray.direction * hitData.t;
    float3 normal = normalize(pos - scene.objects[hitData.index].pos);
    // flip normal if necessary
    normal = dot(normal, ray.direction) < 0 ? normal : -1 * normal;
    
    // direct lighting
    //color += mask * getDirectLighting(scene, normal, pos) * scene.objects[hitData.index].color * 0.5f;
    // hack for aliasing issue with bright light source
    if (n == 0)
      color += clamp(mask * scene.objects[hitData.index].emission, 0.0, 1.0);
    else
      color += mask * scene.objects[hitData.index].emission;
    mask *= scene.objects[hitData.index].color;
    
    // create next ray
    ray.origin = pos + normal * PUSH_RAY_ORIGIN;
    // diffuse bounce
    ray.direction = normalize(getCosineWeightedNormal(normal, randState));
    // makeshift glossy BRDF
    //ray.direction = reflect(ray.direction, normal);
    //ray.direction += 0.01* make_float3(curand_uniform(randState), curand_uniform(randState), curand_uniform(randState)) - 0.005;
    //ay.direction = normalize(ray.direction);

    // record first bounce information
    if(n == 0) {
      L.normal += normal;
      L.albedo += scene.objects[hitData.index].color;
      L.depth += hitData.t;
      // update variances
      var.updateVariance(luminance(normal), NORMAL);
      var.updateVariance(luminance(scene.objects[hitData.index].color), ALBEDO);
      var.updateVariance(hitData.t, DEPTH);
    }
  }

  L.color += color;
  // update color variance with final sample color
  var.updateVariance(luminance(color), COLOR);
}

__global__ void pixel_kernel(OutputBuffer output, curandState* randStates, Scene scene, float3* rayBasis, float3* eyePos, int spp) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int id = x * output.width + y;

  if (x >= output.width || y >= output.height)
    return;
  
  //copy random state to local memory
  curandState localRandState = randStates[id];
  
  // take samples
  float3 color = make_float3(0.0f, 0.0f, 0.0f);
  OnlineVarianceBuffer var;
  TraceOutput L;

  for (int i = 0; i < spp; i++) {
    // determine ray direction by interpolating from basis
    float2 screenPos = make_float2(x, y);
    if (spp != 1) {
      screenPos.x += curand_uniform(&localRandState)*1.0f - 0.5f;
      screenPos.y += curand_uniform(&localRandState)*1.0f - 0.5f;
    }
    screenPos /= make_float2(output.width, output.height);
    Ray ray;
    ray.origin = *eyePos;
    ray.direction = lerp(lerp(rayBasis[0], rayBasis[1], screenPos.y), lerp(rayBasis[2], rayBasis[3], screenPos.y), 1.0f-screenPos.x);
    // trace ray and accumulate color
    trace_ray(L, scene, ray, &localRandState, x, y, var);
  }
  // average over all samples
  L.color /= (float)spp;
  L.normal /= (float)spp;
  L.albedo /= (float)spp;
  L.depth /= (float)spp;
  
  // write to output buffer
  output.buffer[x*output.width*14 + y*14 + 0] = L.color.x;
  output.buffer[x*output.width*14 + y*14 + 1] = L.color.y;
  output.buffer[x*output.width*14 + y*14 + 2] = L.color.z;
  output.buffer[x*output.width*14 + y*14 + 3] = L.normal.x;
  output.buffer[x*output.width*14 + y*14 + 4] = L.normal.y;
  output.buffer[x*output.width*14 + y*14 + 5] = L.normal.z;
  output.buffer[x*output.width*14 + y*14 + 6] = L.albedo.x;
  output.buffer[x*output.width*14 + y*14 + 7] = L.albedo.y;
  output.buffer[x*output.width*14 + y*14 + 8] = L.albedo.z;
  output.buffer[x*output.width*14 + y*14 + 9] = L.depth;
  // get final variances
  output.buffer[x*output.width*14 + y*14 + 10] = var.getVariance(COLOR);
  output.buffer[x*output.width*14 + y*14 + 11] = var.getVariance(NORMAL);
  output.buffer[x*output.width*14 + y*14 + 12] = var.getVariance(ALBEDO);
  output.buffer[x*output.width*14 + y*14 + 13] = var.getVariance(DEPTH);
  // copy rand state back to global memory
  randStates[id] = localRandState;
}

__global__ void setup_random(curandState* states, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int id = x * width + y;
  if (x >= width || y >= height)
    return;
  curand_init(id, 0, 0, &states[id]);
}