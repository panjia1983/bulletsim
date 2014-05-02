#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "bulletrac.h"
#include "utils/logging.h"

namespace py = boost::python;

BOOST_PYTHON_MODULE(cbulletracpy) {
  LoggingInit();
  log4cplus::Logger::getRoot().setLogLevel(GeneralConfig::verbose);
  bt::InitPython();

  py::def("py_tracking", &bt::py_tracking);
}
