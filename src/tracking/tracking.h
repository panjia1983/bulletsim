#pragma once
#include "config_tracking.h"
#include <boost/python.hpp>

namespace bs {
  namespace py = boost::python;

  void InitPython();

  py::object py_tracking(py::object transformer, py::object filtered_clouds, py::object rgb_images, py::object depth_images, int num_iter);

}
