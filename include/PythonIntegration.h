#pragma once
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