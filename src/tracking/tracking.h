#pragma once
#include "config_tracking.h"
#include <boost/python.hpp>

namespace bs {
  namespace py = boost::python;

  void InitPython();

  py::object py_tracking(py::object nodes, float rope_radius, py::object transformer, py::list filtered_clouds, py::object rgb_images, py::object depth_images, int num_iter);
}
