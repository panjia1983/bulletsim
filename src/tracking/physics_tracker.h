#pragma once
#include "tracked_object.h"
#include "sparse_utils.h"
#include "utils/pcl_typedefs.h"
#include "visibility.h"
#include "feature_extractor.h"
#include <cv.h>

class PhysicsTracker {
public:
	typedef boost::shared_ptr<PhysicsTracker> Ptr;

	TrackedObjectFeatureExtractor::Ptr m_objFeatures;
  FeatureExtractor::Ptr m_obsFeatures;

  VisibilityInterface::Ptr m_visInt;

  // latent parameters
  Eigen::MatrixXf m_estPts;
  Eigen::MatrixXf m_stdev;
  Eigen::VectorXf m_priorDist;

  // observed variables
  Eigen::MatrixXf m_obsPts;

  // posterior probabilities
  Eigen::MatrixXf m_pZgivenC;

  // outliers and unmodelled occluders
  Eigen::VectorXf m_outlierDist; //m_obsPts - m_estPts for the fake node responsible for an outlier observation. same for all obsPts.
  Eigen::VectorXf m_vis;

  PhysicsTracker(TrackedObjectFeatureExtractor::Ptr object_features, FeatureExtractor::Ptr observation_features, VisibilityInterface::Ptr visibility_interface);
  PhysicsTracker(TrackedObjectFeatureExtractor::Ptr object_features, FeatureExtractor::Ptr observation_features, VisibilityInterface::Ptr visibility_interface, const Eigen::MatrixXf& stdev);

  void updateFeatures();
  void expectationStep();
  void maximizationStep(bool apply_evidence=true);
};


