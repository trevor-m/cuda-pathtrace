#include "Renderer.h"
#include "Camera.h"
#include "Scene.h"
#include <iostream>
#include <string>
#include "args.hxx"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main(int argc, const char** argv) {
  // set up argument parser
  args::ArgumentParser parser("cuda-pathtrace");
  parser.LongSeparator(" ");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<int> argWidth(parser, "w", "Image/window width (default 512)", {'w', "width"});
  args::ValueFlag<int> argHeight(parser, "h", "Image/window height (default 512)", {'h', "height"});
  args::ValueFlag<int> argSamples(parser, "samples", "Number of samples per pixel (default 4)", {'s', "samples"});
  args::ValueFlag<int> argDevice(parser, "device", "Which CUDA device to use (default 0)", {'d', "device"});
  args::ValueFlag<int> argThreads(parser, "threads", "Number of threads per block (default 8)", {'t', "threads-per-block"});
  args::ValueFlag<std::string> argOutput(parser, "path", "Prefix of output file name(s) (default output/output)", {'o', "output"});
  args::Flag argNoBitmaps(parser, "nobitmap", "Do not output bitmap features - only the exr", {'n', "nobitmap"});
  args::Flag argIteractive(parser, "interactive", "Open in interactive mode  - will only render a single frame if not set", {'i', "interactive"});
  try {
    parser.ParseCLI(argc, argv);
  }
  catch (args::Help) {
    std::cout << parser;
    return 0;
  }
  catch (args::ParseError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }
  catch (args::ValidationError e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }
  //get arguments
  int width = (argWidth) ? args::get(argThreads) : 512;
  int height = (argHeight) ? args::get(argThreads) : 512;
  int threadsPerBlock = (argThreads) ? args::get(argThreads) : 8;
  int samplesPerPixel = (argSamples) ? args::get(argSamples) : 4;
  int cudaDevice = (argDevice) ? args::get(argDevice) : 0;
  std::string outputName = (argOutput) ? args::get(argOutput) : "output/out";
  std::cout << "cuda-pathtrace 0.2" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "Dimensions: " << width << " x " << height << std::endl;
  std::cout << "Threads per block: " << threadsPerBlock << std::endl;
  std::cout << "Samples per pixel: " << samplesPerPixel << std::endl;
  std::cout << "Using CUDA device: " << cudaDevice << std::endl;
  std::cout << "Output file prefix: " << outputName << std::endl;

  // set cuda device
  gpuErrchk(cudaSetDevice(cudaDevice));

  // create renderer
  Scene scene;
  Renderer renderer(width, height, samplesPerPixel, threadsPerBlock);
  Camera camera(glm::vec3(50, 52, 295.6));

  if (argIteractive) {
    // interactive (realtime) mode
    //while(1) {
    //  renderer.Render(camera);
    //  renderer.Denoise();
    //  renderer.CopyBufferToScreen();
    //}
  }
  else {
    // data collection (single frame render) mode
    // render frame
    float renderTime = renderer.Render(scene, camera);
    std::cout << "Render completed in " << renderTime << "ms (" << 1000.0f/renderTime << " fps)" << std::endl;
    // save results
    OutputBuffer buffer(width, height);
    buffer.AllocateCPU();
    buffer.CopyFromGPU(renderer.d_buffer);
    buffer.SaveEXR(outputName);
    if(!argNoBitmaps)
      buffer.SaveBitmaps(outputName);
    buffer.FreeCPU();
  }
  
  return 0;
}
