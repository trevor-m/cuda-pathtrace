![render](https://user-images.githubusercontent.com/12981474/33522069-3971c39c-d798-11e7-8d2b-3825c35ae012.png)
# cuda-pathtrace

cuda-pathtrace is a realtime photorealistic pathtracer implemented in CUDA. It can currently only render diffuse surfaces and scenes containing spheres.

By default, cuda-pathtrace outputs many features which can be used in a denoising algorithm, such as color, surface normals, albedo/texture, depth, along with per-pixel variances for each feature.

cuda-pathtrace's low-sample renders are automatically fed through a deep learning denoising algorithm to provide an interactive realtime photorealistic experience. See the `denoise_cnn/` folder for current status of the denoising algorithm experimentation.

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
* Boost (boost_python, boost_program_options, boost_system)
* Python 2.7
* Pytorch for Python 2.7 and CUDA 8.0

### Compiling

Type `make` to build the application (Tested on Ubuntu 16.04).

## Usage

To launch cuda-pathtrace in interactive (real-time) mode with denoising enabled, use:

```
./pathtrace -i -d
```

To render single frames, cuda-pathtrace accepts the following arguments. The output file <output name>.exr will be created with the results of your render.

```
Options:
  --help                         Print help messages
  -t [ --threads-per-block ] arg Number of threads per block in 2D CUDA
                                 scheduling grid.
  --size arg                     Size of the screen in pixels
  -s [ --samples ] arg           Number of samples per pixel
  --device arg                   Which CUDA device to use for rendering
  -d [ --denoising ]             Use denoising neural network.
  -i [ --interactive ]           Interactive mode - will render single frame
                                 only if not set.
  --nobitmap                     Don't output bitmaps for each channel
  -o [ --output ] arg            Prefix of output file/path
  -x [ --camera-x ] arg          Starting camera position x
  -y [ --camera-y ] arg          Starting camera position y
  -z [ --camera-z ] arg          Starting camera position z
  -c [ --camera-yaw ] arg        Starting camera view yaw
  -p [ --camera-pitch ] arg      Starting camera view pitch
```

## Using the Rendered Features

cuda-pathtrace outputs a multilayered OpenEXR file containing all of the necessary features to train a deep learning denoising algorithm.

### In Python

To load the rendered image and features from the EXR file, the python code in `denoise_cnn/load_data.py` can be used via:

```python
from load_data import load_exr_data

x = load_exr_data("output.exr", preprocess=True)
```

You will need to install the [OpenEXR python bindings](http://www.excamera.com/sphinx/articles-openexr.html). If you are using Windows, I recommened installing from an unofficial [precompiled binary](https://www.lfd.uci.edu/~gohlke/pythonlibs/#openexr) - it will make your life 10x easier.

## Built With
* C++
* CUDA
* glm - Mathmatics functions for Camera class
* GLFW
* OpenGL
* Pytorch
* Python
* Boost
* [tinyexr](https://github.com/syoyo/tinyexr) - Saving OpenEXR files
* [stbimage](https://github.com/nothings/stb) - Saving bitmaps
* Cornell box scene from http://www.kevinbeason.com/smallpt/
