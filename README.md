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

## Using the Rendered Features

cuda-pathtrace outputs a multilayered OpenEXR file containing all of the necessary features to train a deep learning denoising algorithm.

To load the features from the EXR file, the following python code could be used via:
`feaures = load_exr_data("output.exr")`

The following code is untested and is not the most effecient.

```python
def get_layer(infile, layer_name):
  # extract channel names
  channel_names = []
  for layer in infile.header()['channels']:
    if layer_name in layer:
      channel_names.append(layer)
  # make sure we got something
  if len(channel_names) == 0:
    print('Warning: Layer \'%s\' was not found.' % layer_name)
    return None
  # sort to RGB, XYZ, and remove A if more than one channel
  if len(channel_names) > 1:
    channel_names = sorted(channel_names)
    # if RGB, rearrange from BGR to RGB
    if channel_names[0].split('.')[-1] == 'B':
      channel_names = [channel_names[2], channel_names[1], channel_names[0]]
  # get image dimensions
  dw = infile.header()['dataWindow']
  size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
  # get data from channels
  pt = Imath.PixelType(Imath.PixelType.FLOAT)
  data = np.zeros((size[1], size[0], len(channel_names)))
  for i, name in enumerate(channel_names):
    data[:, :, i] = np.fromstring(infile.channel(name, pt), dtype=np.float32).reshape(size[1], size[0])
  # might have to flip if width == height
  if size[0] == size[1]:
    data = np.flipud(data)
  return data

def load_exr_data(filename):
  infile = OpenEXR.InputFile(filename)
  color = get_layer(infile, 'Color')
  normal = get_layer(infile, 'Normal')
  albedo = get_layer(infile, 'Albedo')
  depth = get_layer(infile, 'Depth')
  color_var = get_layer(infile, 'ColorVar')
  normal_var = get_layer(infile, 'NormalVar')
  albedo_var = get_layer(infile, 'AlbedoVar')
  depth_var = get_layer(infile, 'DepthVar')
  # you can also use np.stack to create a 3D array of size [width, height, 14]
  return [color, normal, albedo, depth, color_var, normal_var, albedo_var, depth_var]
```

## Built With
* C++
* CUDA
* [tinyexr](https://github.com/syoyo/tinyexr) - Saving OpenEXR files
* [stbimage](https://github.com/nothings/stb) - Saving bitmaps
