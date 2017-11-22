import OpenEXR
import Imath
import numpy as np

def load_exr_data(filename):
  """Loads a multilayer OpenEXR file and returns a list of all features loaded as numpy arrays"""
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

def get_layer(infile, layer_name):
  # extract channel names
  channel_names = []
  for layer in infile.header()['channels']:
    # add . to end of layer_name so we don't get layers that start with the same prefix too
    if layer_name+'.' in layer:
      channel_names.append(layer)
  # make sure we got something
  if not channel_names:
    print('Warning: Layer \'%s\' was not found.' % layer_name)
    return None
  # sort to RGB, XYZ, and remove A if more than one channel
  if channel_names:
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
  return data