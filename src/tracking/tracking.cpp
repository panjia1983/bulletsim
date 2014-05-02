#include <pcl/ros/conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <cv_bridge/cv_bridge.h>
#include <cv.h>
#include "clouds/utils_ros.h"

#include "clouds/utils_pcl.h"
#include "utils_tracking.h"
#include "utils/logging.h"
#include "utils/utils_vector.h"
#include "visibility.h"
#include "physics_tracker.h"
#include "feature_extractor.h"
#include "initialization.h"
#include "simulation/simplescene.h"
#include "config_tracking.h"
#include "utils/conversions.h"
#include "clouds/cloud_ops.h"
#include "simulation/util.h"
#include "clouds/utils_cv.h"
#include "tracking.h"
#include "algorithm_common.h"


using namespace std;
using namespace Eigen;

namespace cv {
  typedef Vec<uchar, 3> Vec3b;
}

namespace bs
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




  std::vector<std::vector<btVector3> > tracking(const std::vector<btVector3>& nodes, float rope_radius, const btTransform& cam, const std::vector<ColorCloudPtr>& filtered_clouds, std::vector<cv::Mat>& rgb_images, std::vector<cv::Mat>& depth_images, int num_iter)
  {
    CoordinateTransformer transformer(cam);
    std::vector<std::vector<btVector3> > tracking_results;
    GeneralConfig::scale = 100;
    BulletConfig::maxSubSteps = 0;
    BulletConfig::gravity = btVector3(0,0,-0.1);

    cv::Mat mask_image(rgb_images[0].rows, rgb_images[0].cols, CV_8U, true);

    // set up scene
    Scene scene;
    util::setGlobalEnv(scene.env);

    TrackedObject::Ptr trackedObj = callInitServiceAndCreateObject(nodes, rope_radius, filtered_clouds[0], rgb_images[0], mask_image, &transformer);
    if (!trackedObj) throw runtime_error("initialization of object failed.");
    trackedObj->init();
    scene.env->add(trackedObj->m_sim);

    // actual tracking algorithm
    MultiVisibility::Ptr visInterface(new MultiVisibility());
    visInterface->addVisibility(DepthImageVisibility::Ptr(new DepthImageVisibility(&transformer)));

    TrackedObjectFeatureExtractor::Ptr objectFeatures(new TrackedObjectFeatureExtractor(trackedObj));
    CloudFeatureExtractor::Ptr cloudFeatures(new CloudFeatureExtractor());
    PhysicsTracker::Ptr alg(new PhysicsTracker(objectFeatures, cloudFeatures, visInterface));
    PhysicsTrackerVisualizer::Ptr trackingVisualizer(new PhysicsTrackerVisualizer(&scene, alg));

    bool applyEvidence = true;

    for (int n = 0; n < rgb_images.size(); ++n) {
      cloudFeatures->updateInputs(filtered_clouds[n], rgb_images[n], &transformer);
      visInterface->visibilities[0]->updateInput(depth_images[n]);

      for (int i = 0; i < num_iter; ++i) {
        alg->updateFeatures();
        alg->expectationStep();
        alg->maximizationStep(applyEvidence);
        scene.step(.03, 2, .015);   
      }
      
      std::vector<btVector3> nodes = scaleVecs(trackedObj->getPoints(), 1/METERS);
      for (int i = 0; i < nodes.size(); ++i) {
        cout << nodes[i].x() << " " << nodes[i].y() << " " << nodes[i].z() << endl;
      }
      
      tracking_results.push_back(nodes);
    }

    return tracking_results;
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

  std::vector<cv::Mat> fromNdarray4ToRGBImages(py::object a) 
  {
    a = ensureFormat<btScalar>(a);
    py::object shape = a.attr("shape");
    if (py::len(shape) != 4) {
      throw std::runtime_error((boost::format("expected 4-d array, got %d-d instead") % py::len(shape)).str());
    }
    
    size_t out_dim0 = py::extract<size_t>(shape[0]);
    size_t out_dim1 = py::extract<size_t>(shape[1]);
    size_t out_dim2 = py::extract<size_t>(shape[2]);
    size_t out_dim3 = py::extract<size_t>(shape[3]);

    if (out_dim3 != 3) {
      throw std::runtime_error((boost::format("expected shape[3] == 3, got %d instead") % out_dim2).str());
    }

    btScalar* pin = getPointer<btScalar>(a);

    std::vector<cv::Mat> images;

    for (int n = 0; n < out_dim0; ++n) {
      cv::Mat image(out_dim1, out_dim2, CV_8UC3);
      for (int i = 0; i < out_dim1; ++i) {
        for (int j = 0; j < out_dim2; ++j) {
          btScalar* cur = pin + out_dim1 * out_dim2 * out_dim3 * n + out_dim2 * out_dim3 * i + out_dim3 * j;
          image.at<cv::Vec3b>(i, j) = cv::Vec3b( (uchar) (*(cur)), (uchar) (*(cur+1)), (uchar) (*(cur+2)));
        }
      }
    
      images.push_back(image);
    }

    return images;
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

  std::vector<cv::Mat> fromNdarray3ToDepthImages(py::object a)
  {
    a = ensureFormat<btScalar>(a);
    py::object shape = a.attr("shape");
    if (py::len(shape) != 3) {
      throw std::runtime_error((boost::format("expected 3-d array, got %d-d instead") % py::len(shape)).str());
    }
    
    size_t out_dim0 = py::extract<size_t>(shape[0]);
    size_t out_dim1 = py::extract<size_t>(shape[1]);
    size_t out_dim2 = py::extract<size_t>(shape[2]);

    btScalar* pin = getPointer<btScalar>(a);

    std::vector<cv::Mat> images;

    for (int n = 0; n < out_dim0; ++n) {
      cv::Mat image(out_dim1, out_dim2, CV_32FC1);
      for (int i = 0; i < out_dim1; ++i) {
        for (int j = 0; j < out_dim2; ++j) {
          image.at<float>(i, j) = *(pin + out_dim1 * out_dim2 * n + out_dim1 * i + j) / 1000.0;
        }
      }

      images.push_back(image);
    }

    return images;
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

  std::vector<ColorCloud> fromNdarray3ToColorClouds(py::object a)
  {
    a = ensureFormat<btScalar>(a);
    py::object shape = a.attr("shape");
    if (py::len(shape) != 3) {
      throw std::runtime_error((boost::format("expected 3-d array, got %d-d instead") % py::len(shape)).str());
    }
    
    size_t out_dim0 = py::extract<size_t>(shape[0]); // number of clouds
    size_t out_dim1 = py::extract<size_t>(shape[1]); // number of points in each cloud
    size_t out_dim2 = py::extract<size_t>(shape[2]); // dimension (6 for color cloud)

    btScalar* pin = getPointer<btScalar>(a);
    
    std::vector<ColorCloud> color_clouds;

    for (int n = 0; n < out_dim0; ++n) {
      ColorCloud color_cloud;
      color_cloud.points.resize(out_dim1);
    
      for (int i = 0; i < out_dim1; ++i) {
        btScalar* cur = pin + out_dim1 * out_dim2 * n + out_dim1 * i;
        color_cloud.points[i].x = *cur;
        color_cloud.points[i].y = *(cur + 1);
        color_cloud.points[i].z = *(cur + 2);
        color_cloud.points[i].r = *(cur + 3);
        color_cloud.points[i].g = *(cur + 4);
        color_cloud.points[i].b = *(cur + 5);      
      }

      color_clouds.push_back(color_cloud);
    }

    return color_clouds;
  }


  std::vector<ColorCloud> fromNdarray2ListToColorClouds(py::list a)
  {
    std::vector<ColorCloud> color_clouds;

    int n = py::len(a);
    for (int i = 0; i < n; ++i) {
      py::object input = py::extract<py::object>(a[i]);
      ColorCloud output = fromNdarray2ToColorCloud(input);
      color_clouds.push_back(output);
    }

    return color_clouds;
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

  py::object toNdarray3(const vector<vector<btVector3> >& vs) {
    py::object out = numpy.attr("empty")(py::make_tuple(vs.size(), vs[0].size(), 3), type_traits<btScalar>::npname);  
    btScalar* pout = getPointer<btScalar>(out);
    for (int i = 0; i < vs.size(); ++i) {
      size_t len_v = vs[i].size();
      for (int j = 0; j < len_v; ++j) {
        for (int k = 0; k < 3; ++k) {
          *(pout + 3 * len_v * i + 3 * j + k) = vs[i][j].m_floats[k];
        }
      }
    }

    return out;
  }

  vector<btVector3> fromNdarray2ToNodes(py::object a)
  {
    a = ensureFormat<btScalar>(a);
    py::object shape = a.attr("shape");
    if (py::len(shape) != 2) {
      throw std::runtime_error((boost::format("expected 2-d array, got %d-d instead") % py::len(shape)).str());
    }
    
    size_t out_dim0 = py::extract<size_t>(shape[0]); // number of points
    size_t out_dim1 = py::extract<size_t>(shape[1]); // dimension

    btScalar* pin = getPointer<btScalar>(a);
    
    vector<btVector3> nodes;
    for (int i = 0; i < out_dim0; ++i) {
      btScalar* cur = pin + out_dim1 * i;
      btVector3 v(*cur, *(cur+1), *(cur+2));
      nodes.push_back(v);
    }
    
    return nodes;
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


  py::object py_tracking(py::object nodes, float rope_radius, py::object transformer, py::object filtered_clouds, py::object rgb_images, py::object depth_images, int num_iter)
  {
    btTransform transformer_ = toBtTransform(transformer);
    std::vector<cv::Mat> rgb_images_;
    std::vector<cv::Mat> depth_images_;
    std::vector<ColorCloudPtr> filtered_clouds_;
    std::vector<btVector3> nodes_;

    rgb_images_ = fromNdarray4ToRGBImages(rgb_images);
    depth_images_ = fromNdarray3ToDepthImages(depth_images);
    std::vector<ColorCloud> filtered_clouds__ = fromNdarray3ToColorClouds(filtered_clouds);
    nodes_ = fromNdarray2ToNodes(nodes);

    for (int i = 0; i < filtered_clouds__.size(); ++i) {
      ColorCloudPtr filtered_cloud(new ColorCloud(filtered_clouds__[i]));
      filtered_clouds_.push_back(filtered_cloud);
    }
  
    std::vector<std::vector<btVector3> > tracking_results = tracking(nodes_, rope_radius, transformer_, filtered_clouds_, rgb_images_, depth_images_, num_iter);

    return toNdarray3(tracking_results);
  }


  py::object py_tracking2(py::object nodes, float rope_radius, py::object transformer, py::list filtered_clouds, py::object rgb_images, py::object depth_images, int num_iter)
  {
    int dummy_argc=0;
    ros::init((int&)dummy_argc, NULL, "tracking");  
    ros::NodeHandle n;

    btTransform transformer_ = toBtTransform(transformer);
    std::vector<cv::Mat> rgb_images_;
    std::vector<cv::Mat> depth_images_;
    std::vector<ColorCloudPtr> filtered_clouds_;
    std::vector<btVector3> nodes_;

    rgb_images_ = fromNdarray4ToRGBImages(rgb_images);
    depth_images_ = fromNdarray3ToDepthImages(depth_images);
    std::vector<ColorCloud> filtered_clouds__ = fromNdarray2ListToColorClouds(filtered_clouds);
    nodes_ = fromNdarray2ToNodes(nodes);
    
    
    for (int i = 0; i < filtered_clouds__.size(); ++i) {
      ColorCloudPtr filtered_cloud(new ColorCloud(filtered_clouds__[i]));
      filtered_clouds_.push_back(filtered_cloud);
    }

  
    std::vector<std::vector<btVector3> > tracking_results = tracking(nodes_, rope_radius, transformer_, filtered_clouds_, rgb_images_, depth_images_, num_iter);
   
    py::object results = toNdarray3(tracking_results);

    return results;
  }


}
