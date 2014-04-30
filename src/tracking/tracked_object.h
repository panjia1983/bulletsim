#pragma once
#include <Eigen/Dense>
#include <simulation/basicobjects.h>
#include <simulation/rope.h>
#include "sparse_utils.h"
#include "utils/pcl_typedefs.h"
#include "utils/cvmat.h"
#include "config_tracking.h"
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include "utils/utils_pcl.h"

class CoordinateTransformer;

class TrackedObject {
public:
  typedef boost::shared_ptr<TrackedObject> Ptr;
  EnvironmentObject::Ptr m_sim;
  std::string m_type;
  int m_nNodes;
  Eigen::VectorXf m_sigs;
  Eigen::MatrixXf m_colors;

  TrackedObject(EnvironmentObject::Ptr sim, string type);
  TrackedObject(string type);
  void init();

  virtual std::vector<btVector3> getPoints() = 0;
  Eigen::MatrixXf& getColors();
  virtual void initColors() { m_colors = Eigen::MatrixXf::Constant(m_nNodes, 3, 1.0); }

  virtual const Eigen::VectorXf getPriorDist();

  virtual const Eigen::VectorXf getOutlierDist() { return Eigen::VectorXf::Constant(3, TrackingConfig::pointOutlierDist*METERS); }
  virtual const Eigen::VectorXf getOutlierStdev() { return Eigen::VectorXf::Constant(3, TrackingConfig::pointPriorDist*METERS); }
  virtual void applyEvidence(const Eigen::MatrixXf& corr, const Eigen::MatrixXf& obsPts) = 0;
  virtual EnvironmentObject* getSim() { return NULL; }
};

class TrackedRope : public TrackedObject { 
public:
  typedef boost::shared_ptr<TrackedRope> Ptr;

  TrackedRope(CapsuleRope::Ptr sim);

  std::vector<btVector3> getPoints();
  void applyEvidence(const Eigen::MatrixXf& corr, const Eigen::MatrixXf& obsPts);
  CapsuleRope* getSim() {return dynamic_cast<CapsuleRope*>(m_sim.get());}
  void initColors();

protected:
  Eigen::VectorXf m_masses;
};

std::vector<btVector3> calcImpulsesDamped(const std::vector<btVector3>& estPos, const std::vector<btVector3>& estVel,
  const std::vector<btVector3>& obsPts, const Eigen::MatrixXf& corr, const vector<float>& masses, float kp, float kd) ;

