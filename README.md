# cuda-pathtrace
cuda-pathtrace is a realtime photorealistic pathtracer implemented in CUDA.

It can currently only render diffuse surfaces and scenes containing spheres.

By default, cuda-pathtrace outputs the following features which can be used in a denoising algorithm:

* Color
* Normals
* Albedo
* Depth
* Color Variance
* Normal Variance
* Albedo Variance
* Depth Variance

In the future, cuda-pathtrace's renders will be automatically fed through a deep learning denoising algorithm to provide an interactive realtime photorealistic experience.

## Installation

### Prerequisites

You will need the following in order to build the application:

* [NVIDIA CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-downloads)
* [glm](https://glm.g-truc.net/0.9.8/index.html) - Just copy the glm folder to /usr/include

### Compiling

Type `make` to build the application (Tested on Ubuntu 16.04).

### Usage

Use one of the following to render the scene. The output file <outpt name>.exr will be created with the results of your render.

`pathtrace <samples per pixel>`

`pathtrace <samples per pixel> <output name>`

`pathtrace <samples per pixel> <output name> <CUDA device>`

## Built With
* C++
* CUDA
* [tinyexr](https://github.com/syoyo/tinyexr) - Saving OpenEXR files
* [stbimage](https://github.com/nothings/stb) - Saving bitmaps
