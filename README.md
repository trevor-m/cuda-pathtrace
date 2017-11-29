![render](https://user-images.githubusercontent.com/12981474/32823079-768c3b56-c990-11e7-85d5-9b55fcb8572e.png)
# cuda-pathtrace

cuda-pathtrace is a realtime photorealistic pathtracer implemented in CUDA. It can currently only render diffuse surfaces and scenes containing spheres.

By default, cuda-pathtrace outputs many features which can be used in a denoising algorithm, such as color, surface normals, albedo/texture, depth, along with per-pixel variances for each feature.

In the future, cuda-pathtrace's low-sample renders will be automatically fed through a deep learning denoising algorithm to provide an interactive realtime photorealistic experience. See the `denoising/` folder for current status of the denoising algorithm experimentation.

## Installation

### Prerequisites

You will need the following in order to build the application:

* [NVIDIA CUDA Toolkit 8.0](https://developer.nvidia.com/cuda-downloads)
* [glm](https://glm.g-truc.net/0.9.8/index.html) 
  * Just copy the glm folder to your include path (/usr/include)
* [GLFW](https://github.com/glfw/glfw)
  * Clone the repo and cd to the folder
  * Run `cmake -DBUILD_SHARED_LIBS=ON .`
  * Run `make`
  * Run `sudo make install`
  * Copy /usr/local/lib/libglfw.so.3 to your cuda-pathtrace repo, or add /usr/local/lib/ to your LD_LIBRARY_PATH

### Compiling

Type `make` to build the application (Tested on Ubuntu 16.04).

## Usage

To launch cuda-pathtrace in interactive (real-time) mode, use:

```
./pathtrace -i
```

To render cuda-pathtrace accepts the following arguments. The output file <output name>.exr will be created with the results of your render.

```
  ./pathtrace {OPTIONS}

    cuda-pathtrace

  OPTIONS:

      -h, --help                        Display this help menu
      -w [w], --width [w]                Image/window width and height (default 512)
      -s [samples], --samples [samples]  Number of samples per pixel (default 4)
      -d [device], --device [device]     Which CUDA device to use (default 0)
      -t [threads], --threads-per-block [threads]                         
                                         Number of threads per block (default 8)
      -x [x], --camera-x [x]             Starting camera position x
      -y [y], --camera-y [y]             Starting camera position y
      -z [z], --camera-z [z]             Starting camera position z
      -c [yaw], --camera-yaw [yaw]       Starting camera view yaw
      -p [pitch], --camera-pitch [pitch] Starting camera view pitch
      -o [path], --output [path]         Prefix of output file name(s) (default
                                        output/output)
      -n, --nobitmap                    Do not output bitmap features - only the
                                        exr
      -i, --interactive                 Open in interactive mode - will only
                                        render a single frame if not set


```

## Using the Rendered Features

cuda-pathtrace outputs a multilayered OpenEXR file containing all of the necessary features to train a deep learning denoising algorithm.

### In Python

To load the rendered image and features from the EXR file, the python code in `denoising/load_data.py` can be used via:

```python
from load_data import load_exr_data

x = load_exr_data("output.exr")
```

You will need to install the [OpenEXR python bindings](http://www.excamera.com/sphinx/articles-openexr.html). If you are Windows, I recommened installing from an unofficial [precompiled binary](https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr) - it will make your life 10x easier.

## Built With
* C++
* CUDA
* glm - Mathmatics functions for Camera class
* GLFW
* OpenGL
* [tinyexr](https://github.com/syoyo/tinyexr) - Saving OpenEXR files
* [stbimage](https://github.com/nothings/stb) - Saving bitmaps
* [args](https://github.com/Taywee/args) - Parsing command line arguments
* Cornell box scene from http://www.kevinbeason.com/smallpt/
