#include "Renderer.h"
#include "Denoiser.h"
#include "Camera.h"
#include "Window.h"
#include "Scene.h"
#include <iostream>
#include <string>
#include <cstdlib>

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
  //get arguments
  int width =  512;
  int height = width;
  int threadsPerBlock = 8;
  int samplesPerPixel = 2;
  int cudaDevice = 0;
  // camera arguments
  float cameraPos[3] = { 50.0f, 52.0f, 295.6f };
  float cameraView[2] = {-90.0f, 0.0f};
  bool denoising = true;
  std::string outputName = "output/out";
  std::cout << "cuda-pathtrace 0.2" << std::endl;
  std::cout << "------------------" << std::endl;
  std::cout << "Dimensions: " << width << " x " << height << std::endl;
  std::cout << "Threads per block: " << threadsPerBlock << std::endl;
  std::cout << "Samples per pixel: " << samplesPerPixel << std::endl;
  std::cout << "Using CUDA device: " << cudaDevice << std::endl;
  bool argInteractive = true;
  if (!argInteractive)
    std::cout << "Output file prefix: " << outputName << std::endl;
  std::cout << "Camera: " << cameraPos[0] << " " << cameraPos[1] << " " << cameraPos[2] << " " << cameraView[0] << " " << cameraView[1] << std::endl;

  // set cuda device
  gpuErrchk(cudaSetDevice(cudaDevice));
  //if (argIteractive)
  //   gpuErrchk(cudaGLSetGLDevice(cudaDevice));

  setenv("PYTHONPATH", "./denoise_cnn", 1);

  Py_Initialize();
  PyEval_InitThreads();
  long _tensor_ptr = -1;
  py::dict locals;
  PythonState state;
  //py::object attributeError = state.import("exceptions").attr("AttributeError");
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
    buffer.SaveBitmaps(outputName);
    buffer.FreeCPU();
  }
  
  return 0;
}