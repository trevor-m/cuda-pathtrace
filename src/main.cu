#include "Renderer.h"
#include "Denoiser.h"
#include "Camera.h"
#include "Window.h"
#include "Scene.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include "PythonIntegration.h"
#include "boost/program_options.hpp"
namespace po = boost::program_options;

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, const char** argv) {
  // default arguments
  int size =  512;
  int threadsPerBlock = 8;
  int samplesPerPixel = 4;
  int cudaDevice = 0;
  float cameraPos[3] = { 50.0f, 52.0f, 295.6f };
  float cameraView[2] = {-90.0f, 0.0f};
  bool denoising = false;
  bool interactive = false;
  bool noBitmap = false;
  std::string outputName = "output/out";
  //get arguments
  po::options_description desc("Options"); 
  desc.add_options() 
    ("help,h", "Print help messages") 
    ("threads-per-block,t", po::value<int>(&threadsPerBlock), "Number of threads per block in 2D CUDA scheduling grid.") 
    ("size", po::value<int>(&size), "Size of the screen in pixels")
    ("samples,s", po::value<int>(&samplesPerPixel), "Number of samples per pixel")
    ("device", po::value<int>(&cudaDevice), "Which CUDA device to use for rendering")
    ("denoising,d", "Use denoising neural network.")
    ("interactive,i", "Interactive mode - will render single frame only if not set.")
    ("nobitmap", "Don't output bitmaps for each channel")
    ("output,o", po::value<std::string>(&outputName), "Prefix of output file/path")
    ("camera-x,x", po::value<float>(&cameraPos[0]), "Starting camera position x")
    ("camera-y,y", po::value<float>(&cameraPos[1]), "Starting camera position y")
    ("camera-z,z", po::value<float>(&cameraPos[2]), "Starting camera position z")
    ("camera-yaw,c", po::value<float>(&cameraView[0]), "Starting camera view yaw")
    ("camera-pitch,p", po::value<float>(&cameraView[1]), "Starting camera view pitch");
  po::variables_map vm; 
  try { 
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if ( vm.count("help")  ) 
    { 
      std::cout << "cuda-pathtrace" << std::endl 
                << desc << std::endl; 
      return 0; 
    }
    po::notify(vm);
  } 
  catch(po::error& e) { 
    std::cerr << "ERROR: " << e.what() << std::endl << std::endl; 
    std::cerr << desc << std::endl; 
    return 1; 
  }
  denoising = vm.count("denoising");
  interactive = vm.count("interactive");
  noBitmap = vm.count("nobitmap");
  int width, height;
  width = height = size;
  std::cout << "cuda-pathtrace 0.3" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "Dimensions: " << width << " x " << height << std::endl;
  std::cout << "Threads per block: " << threadsPerBlock << std::endl;
  std::cout << "Samples per pixel: " << samplesPerPixel << std::endl;
  std::cout << "Using CUDA device: " << cudaDevice << std::endl;
  if (!interactive)
    std::cout << "Output file prefix: " << outputName << std::endl;
  else {
    std::cout << "Running in interactive mode: ";
    if (denoising)
      std::cout << "denoising is on" << std::endl;
    else
      std::cout << "denoising is off" << std::endl;
  }
  std::cout << "Camera: " << cameraPos[0] << " " << cameraPos[1] << " " << cameraPos[2] << " " << cameraView[0] << " " << cameraView[1] << std::endl;

  // set cuda device
  gpuErrchk(cudaSetDevice(cudaDevice));
  //if (argIteractive)
  //   gpuErrchk(cudaGLSetGLDevice(cudaDevice));

  void* tensor_ptr = NULL;
  py::object modify_tensor, tensor;
  if (denoising) {
    // intialize python
    setenv("PYTHONPATH", "./denoise_cnn", 1);
    Py_Initialize();
    PyEval_InitThreads();
    long _tensor_ptr = -1;
    py::dict locals;
    PythonState state;
    state.exec("import torch\n"
                "from train import test, load_pretrained\n"
                "model = load_pretrained()\n"
                "def make_tensor():\n"
                "    return torch.cuda.FloatTensor(512, 512, 14)\n"
                "def modify_tensor(tensor):\n"
                "    tensor[:,:,:3] = test(model, tensor)[:, :, :]\n"
              );
    py::object torch = state.globals["torch"];
    py::object attributeError = py::import("exceptions").attr("AttributeError");
    py::object make_tensor = state.globals["make_tensor"];
    modify_tensor = state.globals["modify_tensor"];
    tensor = make_tensor();
    locals["t"] = tensor;
    state.exec("data_ptr = t.data_ptr()", locals);
    _tensor_ptr = py::extract<long>(locals["data_ptr"]);
    std::cout << "init torch.cuda.FloatTensor=" << _tensor_ptr << std::endl;
    if (_tensor_ptr < 0) {
      std::cout << "Error allocating torch.cuda.FloatTensor" << std::endl;
        return 0;
    }
    tensor_ptr = reinterpret_cast<void*>(_tensor_ptr);
  }

  // load scene and create renderer
  Scene scene;
  Renderer renderer(width, height, samplesPerPixel, threadsPerBlock);
  Denoiser denoiser(width, height, threadsPerBlock);
  Camera camera(glm::vec3(cameraPos[0], cameraPos[1], cameraPos[2]), cameraView[0], cameraView[1]);

  // allocate output buffer
  OutputBuffer d_buffer(width, height);
  if (denoising) {
    // torch tensor
    d_buffer.buffer = (float*)tensor_ptr;
  }
  else {
    // regular cuda memory
    d_buffer.AllocateGPU();
  }

  if (interactive) {
    // interactive (realtime) mode
    Window window(width, height, &camera, &denoising);
    GLPixelBuffer denoisedBuffer(width, height);

    while(!window.ShouldClose()) {
      window.DoMovement();
      renderer.Render(d_buffer, scene, camera);
      if (denoising) {
        try {
          modify_tensor(tensor);
        }
        catch (const py::error_already_set&) {
          PyObject *ptype, *pvalue, *ptraceback;
          PyErr_Fetch(&ptype, &pvalue, &ptraceback);

          py::handle<> hType(ptype);
          py::object extype(hType);
          py::handle<> hTraceback(ptraceback);
          py::object traceback(hTraceback);

          //Extract error message
          std::string strErrorMessage = py::extract<std::string>(pvalue);

          //Extract line number (top entry of call stack)
          // if you want to extract another levels of call stack
          // also process traceback.attr("tb_next") recurently
          long lineno =py::extract<long> (traceback.attr("tb_lineno"));
          std::string filename = py::extract<std::string>(traceback.attr("tb_frame").attr("f_code").attr("co_filename"));
          std::string funcname = py::extract<std::string>(traceback.attr("tb_frame").attr("f_code").attr("co_name"));
          std::cout << strErrorMessage << std::endl << "line: " << lineno << std::endl;
          std::cout << filename << " " << funcname << std::endl;
        }
      }
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
    if (!noBitmap)
      buffer.SaveBitmaps(outputName);
    buffer.FreeCPU();
  }
  
  return 0;
}