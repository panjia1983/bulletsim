#pragma once
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <cv.h>

#include "utils/utils_pcl.h"
#include "utils_tracking.h"
#include "utils/logging.h"
#include "utils/utils_vector.h"
#include "visibility.h"
#include "physics_tracker.h"
#include "feature_extractor.h"
//#include "initialization.h"
#include "config_tracking.h"
#include "utils/conversions.h"
#include "simulation/util.h"
#include "simulation/bulletsim_lite.h"
#include "simulation/environment.h"

using namespace std;
using namespace Eigen;

namespace bt {
  namespace py = boost::python;

  class BULLETSIM_API TrackedRope : public TrackedObject {
  public:
    typedef boost::shared_ptr<TrackedRope> Ptr;
    
    TrackedRope(bs::CapsuleRopePtr sim, const Vector3f& default_color);
    
    std::vector<btVector3> getPoints();
    std::vector<btVector3> getNodes();
    void applyEvidence(const Eigen::MatrixXf& corr, const Eigen::MatrixXf& obsPts);
    void initColors();

  protected:
    bs::CapsuleRopePtr m_sim;
    Vector3f m_default_color;
  };

  void InitPython();

  //py::object BULLETSIM_API py_tracking(bs::CapsuleRopePtr sim, bs::BulletEnvironmentPtr env, py::object cam, py::object cloud, py::object rgb_image, py::object depth_image, int num_iter);

  py::list py_tracking(bs::CapsuleRopePtr sim, bs::BulletEnvironmentPtr env, py::object cam, py::object cloud, py::object rgb_image, py::object depth_image, int num_iter, py::object stdev);


  
}
