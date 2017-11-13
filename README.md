![out8192_color](https://user-images.githubusercontent.com/12981474/32717689-6e905072-c80f-11e7-9e9b-bf5d44ae011f.png)
# cuda-pathtrace
cuda-pathtrace is a realtime photorealistic pathtracer implemented in CUDA. It can currently only render diffuse surfaces and scenes containing spheres.

By default, cuda-pathtrace outputs many features which can be used in a denoising algorithm, such as color, surface normals, albedo/texture, depth, along with per-pixel variances for each feature.

In the future, cuda-pathtrace's low-sample renders will be automatically fed through a deep learning denoising algorithm to provide an interactive realtime photorealistic experience.

## Installation

### Prerequisites

You will need the following in order to build the application:

* [NVIDIA CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-downloads)
* [glm](https://glm.g-truc.net/0.9.8/index.html) - Just copy the glm folder to /usr/include

### Compiling

Type `make` to build the application (Tested on Ubuntu 16.04).

## Usage

Use one of the following commands to render the scene. The output file <outpt name>.exr will be created with the results of your render.

`./pathtrace <samples per pixel>`

`./pathtrace <samples per pixel> <output name>`

`./pathtrace <samples per pixel> <output name> <CUDA device>`

## Built With
* C++
* CUDA
* [tinyexr](https://github.com/syoyo/tinyexr) - Saving OpenEXR files
* [stbimage](https://github.com/nothings/stb) - Saving bitmaps
