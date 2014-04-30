#pragma once
#include "tracked_object.h"
#include "utils/pcl_typedefs.h"
#include "visibility.h"
#include "algorithm_common.h"
#include "feature_extractor.h"

class SimplePhysicsTracker {
public:
  
  typedef boost::shared_ptr<SimplePhysicsTracker> Ptr;
  
  Environment::Ptr m_env;
  TrackedObject::Ptr m_obj;
  VisibilityInterface::Ptr m_visInt;
  TrackedObjectFeatureExtractor::Ptr m_obj_features;
  CloudFeatureExtractor::Ptr m_cloud_features;

  Eigen::MatrixXf m_estPts;
  Eigen::MatrixXf m_obsPts;
  Eigen::MatrixXf m_stdev;
  Eigen::VectorXf m_prior_dist;
  Eigen::VectorXf m_outlier_dist; //m_obsPts - m_estPts for the fake node responsible for an outlier observation. same for all obsPts.
  Eigen::MatrixXf m_obsDebug;

  ColorCloudPtr m_obsCloud;

  bool m_applyEvidence;
  int m_count;

  SimplePhysicsTracker(TrackedObject::Ptr, VisibilityInterface::Ptr, Environment::Ptr);
  void updateInput(ColorCloudPtr filteredCloud);
  void doIteration();
};
