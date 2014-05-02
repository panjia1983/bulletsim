#include "bulletrac.h"
#include "algorithm_common.h"

namespace bt
{
  static py::object openravepy, numpy;

  void InitPython() {
    openravepy = py::import("openravepy");
    numpy = py::import("numpy");
  }

  template<typename T>
  struct type_traits {
    static const char* npname;
  };
  template<> const char* type_traits<float>::npname = "float32";
  template<> const char* type_traits<int>::npname = "int32";
  template<> const char* type_traits<double>::npname = "float64";

  template <typename T>
  T* getPointer(const py::object& arr) {
    long int i = py::extract<long int>(arr.attr("ctypes").attr("data"));
    T* p = (T*)i;
    return p;
  }

  template<typename T>
  py::object ensureFormat(py::object ndarray) {
    // ensure C-order and data type, possibly making a new ndarray
    return numpy.attr("ascontiguousarray")(ndarray, type_traits<T>::npname);
  }

  template<typename T>
  void fromNdarray2(py::object a, vector<T> &out, size_t &out_dim0, size_t &out_dim1) {
    a = ensureFormat<T>(a);
    py::object shape = a.attr("shape");
    if (py::len(shape) != 2) {
      throw std::runtime_error((boost::format("expected 2-d array, got %d-d instead") % py::len(shape)).str());
    }
    out_dim0 = py::extract<size_t>(shape[0]);
    out_dim1 = py::extract<size_t>(shape[1]);
    out.resize(out_dim0 * out_dim1);
    memcpy(out.data(), getPointer<T>(a), out_dim0*out_dim1*sizeof(T));
  }

  
  py::object toNdarray2(const vector<btVector3> &vs) {
    py::object out = numpy.attr("empty")(py::make_tuple(vs.size(), 3), type_traits<btScalar>::npname);
    btScalar* pout = getPointer<btScalar>(out);
    for (int i = 0; i < vs.size(); ++i) {
      for (int j = 0; j < 3; ++j) {
        *(pout + 3*i + j) = vs[i].m_floats[j];
      }
    }
    return out;
  }

  btTransform toBtTransform(py::object py_hmat, btScalar scale=1) {
    vector<btScalar> hmat; size_t dim0, dim1;
    fromNdarray2(py_hmat.attr("T"), hmat, dim0, dim1);
    if (dim0 != 4 || dim1 != 4) {
      throw std::runtime_error((boost::format("expected 4x4 matrix, got %dx%d") % dim0 % dim1).str());
    }
    btTransform t;
    t.setFromOpenGLMatrix(hmat.data());
    t.getOrigin() *= scale;
    return t;
  }

  Vector3f averageColor(ColorCloudPtr cloud)
  {
    Vector3f rgb(0.0, 0.0, 0.0);
    for (int i = 0; i < cloud->points.size(); ++i)
    {
      rgb += Vector3f(cloud->points[i].r, cloud->points[i].g, cloud->points[i].b);
    }

    rgb = rgb / (float)cloud->points.size();
    return rgb;
  }



  TrackedRope::TrackedRope(bs::CapsuleRopePtr sim, const Vector3f& default_color) : TrackedObject("rope")
  {
    m_sim = sim;
    m_default_color = default_color;
    m_nNodes = sim->GetNodes().size();
  }

  std::vector<btVector3> TrackedRope::getPoints()
  {
    return m_sim->GetNodes();
  }
  
  void TrackedRope::applyEvidence(const Eigen::MatrixXf& corr, const Eigen::MatrixXf& obsPts)
  {
    vector<btVector3> estPos = m_sim->GetNodes();
    vector<btVector3> estVel = m_sim->GetLinearVelocities();
    vector<float> masses = m_sim->GetMasses();

    vector<btVector3> impulses = calcImpulsesDamped(estPos, estVel, toBulletVectors(FE::activeFeatures2Feature(obsPts, FE::FT_XYZ)), corr, masses, TrackingConfig::kp_rope, TrackingConfig::kd_rope);

    m_sim->ApplyCentralImpulses(impulses);
  }

  void TrackedRope::initColors() {
    m_colors.resize(m_nNodes, 3);
    for (int i = 0; i < m_nNodes; ++i) {
      m_colors.row(i) = m_default_color.transpose();
    }
  }

  std::vector<btVector3> tracking(bs::CapsuleRopePtr sim, bs::BulletEnvironmentPtr env, const btTransform& cam, ColorCloudPtr cloud, cv::Mat rgb_image, cv::Mat depth_image, int num_iter)
  {
    Vector3f rope_color = averageColor(cloud);
    TrackedRope::Ptr rope(new TrackedRope(sim, rope_color));
    rope->init();
    MultiVisibility::Ptr visInterface(new MultiVisibility());
    CoordinateTransformer transformer(cam);
    visInterface->addVisibility(DepthImageVisibility::Ptr(new DepthImageVisibility(&transformer)));
    TrackedObjectFeatureExtractor::Ptr objectFeatures(new TrackedObjectFeatureExtractor(rope));
    CloudFeatureExtractor::Ptr cloudFeatures(new CloudFeatureExtractor());
    PhysicsTracker::Ptr alg(new PhysicsTracker(objectFeatures, cloudFeatures, visInterface));
    
    bool applyEvidence = true;
    for (int i = 0; i < num_iter; ++i)
    {
      cloudFeatures->updateInputs(cloud, rgb_image, &transformer);
      visInterface->visibilities[0]->updateInput(depth_image);
      alg->updateFeatures();
      alg->expectationStep();
      alg->maximizationStep(applyEvidence);
      env->Step(.03, 2, .015);
    }
    
    std::vector<btVector3> nodes = scaleVecs(rope->getPoints(), 1);
    
    return nodes;
  }  

  cv::Mat fromNdarray3ToRGBImage(py::object a)
  {
    a = ensureFormat<btScalar>(a);
    py::object shape = a.attr("shape");
    if (py::len(shape) != 3) {
      throw std::runtime_error((boost::format("expected 3-d array, got %d-d instead") % py::len(shape)).str());
    }
    
    size_t out_dim0 = py::extract<size_t>(shape[0]);
    size_t out_dim1 = py::extract<size_t>(shape[1]);
    size_t out_dim2 = py::extract<size_t>(shape[2]);

    if (out_dim2 != 3) {
      throw std::runtime_error((boost::format("expected shape[2] == 3, got %d instead") % out_dim2).str());
    }

    btScalar* pin = getPointer<btScalar>(a);
    cv::Mat image(out_dim0, out_dim1, CV_8UC3);
    for (int i = 0; i < out_dim0; ++i) {
      for (int j = 0; j < out_dim1; ++j) {
        btScalar* cur = pin + out_dim1 * out_dim2 * i + out_dim2 * j;
        image.at<cv::Vec3b>(i, j) = cv::Vec3b( (uchar) (*(cur)), (uchar) (*(cur+1)), (uchar) (*(cur+2)));
      }
    }

    return image;
  }


  cv::Mat fromNdarray2ToDepthImage(py::object a)
  {
    a = ensureFormat<btScalar>(a);
    py::object shape = a.attr("shape");
    if (py::len(shape) != 2) {
      throw std::runtime_error((boost::format("expected 2-d array, got %d-d instead") % py::len(shape)).str());
    }
    
    size_t out_dim0 = py::extract<size_t>(shape[0]);
    size_t out_dim1 = py::extract<size_t>(shape[1]);

    btScalar* pin = getPointer<btScalar>(a);
    cv::Mat image(out_dim0, out_dim1, CV_32FC1);
    for (int i = 0; i < out_dim0; ++i) {
      for (int j = 0; j < out_dim1; ++j) {
        image.at<float>(i, j) = *(pin + out_dim1 * i + j) / 1000.0;
      }
    }

    return image;
  }


  ColorCloud fromNdarray2ToColorCloud(py::object a)
  {
    a = ensureFormat<btScalar>(a);
    py::object shape = a.attr("shape");
    if (py::len(shape) != 2) {
      throw std::runtime_error((boost::format("expected 2-d array, got %d-d instead") % py::len(shape)).str());
    }
    
    size_t out_dim0 = py::extract<size_t>(shape[0]); // number of points
    size_t out_dim1 = py::extract<size_t>(shape[1]); // dimension

    btScalar* pin = getPointer<btScalar>(a);
    
    ColorCloud color_cloud;
    color_cloud.points.resize(out_dim0);

    for (int i = 0; i < out_dim0; ++i) {
      btScalar* cur = pin + out_dim1 * i;
      color_cloud.points[i].x = *cur;
      color_cloud.points[i].y = *(cur + 1);
      color_cloud.points[i].z = *(cur + 2);
      color_cloud.points[i].r = *(cur + 3);
      color_cloud.points[i].g = *(cur + 4);
      color_cloud.points[i].b = *(cur + 5);
    }

    return color_cloud;
  }


  py::object py_tracking(bs::CapsuleRopePtr sim, bs::BulletEnvironmentPtr env, py::object cam, py::object cloud, py::object rgb_image, py::object depth_image, int num_iter)
  {
    
    btTransform cam_ = toBtTransform(cam);
    cv::Mat rgb_image_ = fromNdarray3ToRGBImage(rgb_image);
    
    cv::Mat depth_image_ = fromNdarray2ToDepthImage(depth_image);
    ColorCloudPtr cloud_(new ColorCloud(fromNdarray2ToColorCloud(cloud)));
    std::vector<btVector3> nodes = tracking(sim, env, cam_, cloud_, rgb_image_, depth_image_, num_iter);
    return toNdarray2(nodes);
  }

}


