#include "Renderer.h"
#include "Denoiser.h"
#include "Camera.h"
#include "Window.h"
#include "Scene.h"
#include <iostream>
#include <string>
#include "args.hxx"

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#include <boost/filesystem.hpp>
#include <boost/python.hpp>

namespace py = boost::python;
namespace fs = boost::filesystem;

struct PythonState {
    py::object main_module;
    py::object globals;

    PythonState()
        : main_module(py::object(
              py::handle<>(py::borrowed(PyImport_AddModule("__main__")))))
    {
        globals = main_module.attr("__dict__");
    }

    py::object import(const std::string& module_path)
    {
        return _import(fs::path(module_path));
    }

    py::object _import(const fs::path& module_path)
    {
        try {
            py::dict locals;
            locals["mname"] = module_path.stem().string();
            locals["filename"] = module_path.string();
            py::exec("import importlib.util\n"
                     "spec = importlib.util.spec_from_file_location(mname, "
                     "filename)\n"
                     "imported = importlib.util.module_from_spec(spec)\n"
                     "spec.loader.exec_module(imported)",
                globals, locals);
            return locals["imported"];
        } catch (py::error_already_set& err) {
            PyErr_Print();
        }
        return py::object();
    }

    py::object exec(const char* code, py::dict& locals)
    {
        try {
            return py::exec(code, globals, locals);
        } catch (py::error_already_set& err) {
            PyErr_Print();
        }
        return py::object();
    }

    py::object exec(const char* code)
    {
        try {
            return py::exec(code, globals, globals);
        } catch (py::error_already_set& err) {
            PyErr_Print();
        }
        return py::object();
    }
};


int main(int argc, const char** argv) {
  // set up argument parser
  args::ArgumentParser parser("cuda-pathtrace");
  parser.LongSeparator(" ");
  args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});
  args::ValueFlag<int> argWidth(parser, "w", "Image/window width (default 512)", {'w', "width"});
  args::ValueFlag<int> argSamples(parser, "samples", "Number of samples per pixel (default 4)", {'s', "samples"});
  args::ValueFlag<int> argDevice(parser, "device", "Which CUDA device to use (default 0)", {'d', "device"});
  args::ValueFlag<int> argThreads(parser, "threads", "Number of threads per block (default 8)", {'t', "threads-per-block"});
  args::ValueFlag<float> argCameraX(parser, "x", "Starting camera position x", {'x', "camera-x"});
  args::ValueFlag<float> argCameraY(parser, "y", "Starting camera position y", {'y', "camera-y"});
  args::ValueFlag<float> argCameraZ(parser, "z", "Starting camera position z", {'z', "camera-z"});
  args::ValueFlag<float> argViewYaw(parser, "yaw", "Starting camera view yaw", {'c', "camera-yaw"});
  args::ValueFlag<float> argViewPitch(parser, "pitch", "Starting camera view pitch", {'p', "camera-pitch"});
  args::ValueFlag<std::string> argOutput(parser, "path", "Prefix of output file name(s) (default output/output)", {'o', "output"});
  args::Flag argNoBitmaps(parser, "nobitmap", "Do not output bitmap features - only the exr", {'n', "nobitmap"});
  args::Flag argInteractive(parser, "interactive", "Open in interactive mode  - will only render a single frame if not set", {'i', "interactive"});
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
  int width = (argWidth) ? args::get(argWidth) : 512;
  int height = width;
  int threadsPerBlock = (argThreads) ? args::get(argThreads) : 8;
  int samplesPerPixel = (argSamples) ? args::get(argSamples) : 4;
  int cudaDevice = (argDevice) ? args::get(argDevice) : 0;
  // camera arguments
  float cameraPos[3] = { 50.0f, 52.0f, 295.6f };
  if (argCameraX) 
    cameraPos[0] = args::get(argCameraX);
  if (argCameraY) 
    cameraPos[1] = args::get(argCameraY);
  if (argCameraZ) 
    cameraPos[2] = args::get(argCameraZ);
  float cameraView[2] = {-90.0f, 0.0f};
  if (argViewYaw)
    cameraView[0] = args::get(argViewYaw);
  if (argViewPitch)
    cameraView[1] = args::get(argViewPitch);
  bool denoising = true;
  std::string outputName = (argOutput) ? args::get(argOutput) : "output/out";
  std::cout << "cuda-pathtrace 0.2" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "Dimensions: " << width << " x " << height << std::endl;
  std::cout << "Threads per block: " << threadsPerBlock << std::endl;
  std::cout << "Samples per pixel: " << samplesPerPixel << std::endl;
  std::cout << "Using CUDA device: " << cudaDevice << std::endl;
  if (!argInteractive)
    std::cout << "Output file prefix: " << outputName << std::endl;
  std::cout << "Camera: " << cameraPos[0] << " " << cameraPos[1] << " " << cameraPos[2] << " " << cameraView[0] << " " << cameraView[1] << std::endl;

  // set cuda device
  gpuErrchk(cudaSetDevice(cudaDevice));
  //if (argIteractive)
  //   gpuErrchk(cudaGLSetGLDevice(cudaDevice));

  Py_Initialize();
  PyEval_InitThreads();
  long _tensor_ptr = -1;
  py::dict locals;
  PythonState state;

  state.exec("import torch\n"
              "def make_tensor():\n"
              "    return torch.cuda.FloatTensor(512, 512, 14)\n"
              "def modify_tensor(tensor):\n"
              "    tensor[:, :, 0] = 1.0");

  py::object torch = state.globals["torch"];

  py::object make_tensor = state.globals["make_tensor"];
  py::object modify_tensor = state.globals["modify_tensor"];

  py::object tensor = make_tensor();

  locals["t"] = tensor;

  state.exec("data_ptr = t.data_ptr()", locals);

  _tensor_ptr = py::extract<long>(locals["data_ptr"]);

  std::cout << "init torch.cuda.FloatTensor=" << _tensor_ptr
            << std::endl;

  if (_tensor_ptr < 0) {
      return 0;
  }

  void* tensor_ptr = reinterpret_cast<void*>(_tensor_ptr);

  // load scene and create renderer
  Scene scene;
  Renderer renderer(width, height, samplesPerPixel, threadsPerBlock);
  Denoiser denoiser(width, height, threadsPerBlock);
  Camera camera(glm::make_vec3(cameraPos), cameraView[0], cameraView[1]);
  
  // allocate output buffer
  OutputBuffer d_buffer(width, height);
  if (true) {
    // torch tensor
    d_buffer.buffer = (float*)tensor_ptr;
  }
  else {
    // regular cuda memory
    d_buffer.AllocateGPU();
  }

  if (argInteractive) {
    // interactive (realtime) mode
    Window window(width, height, &camera);
    GLPixelBuffer denoisedBuffer(width, height);

    while(!window.ShouldClose()) {
      window.DoMovement();
      renderer.Render(d_buffer, scene, camera);
      modify_tensor(tensor);
      denoiser.Denoise(d_buffer, denoisedBuffer);
      window.DrawToScreen(denoisedBuffer);
    }
  }
  else {
    // data collection (single frame render) mode
    // render frame
    float renderTime = renderer.Render(d_buffer, scene, camera);
    std::cout << "Render completed in " << renderTime << "ms (" << 1000.0f/renderTime << " fps)" << std::endl;
    std::cout << std::endl;
    // save results
    OutputBuffer buffer(width, height);
    buffer.AllocateCPU();
    buffer.CopyFromGPU(d_buffer);
    buffer.SaveEXR(outputName+".exr");
    if(!argNoBitmaps)
      buffer.SaveBitmaps(outputName);
    buffer.FreeCPU();
  }
  
  return 0;
}