#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "utils/logging.h"
#include "tracking.h"

namespace py = boost::python;

BOOST_PYTHON_MODULE(cbulletracpy2) {
  LoggingInit();
  log4cplus::Logger::getRoot().setLogLevel(GeneralConfig::verbose);

  bs::InitPython();

  py::def("py_tracking", bs::py_tracking);
}
