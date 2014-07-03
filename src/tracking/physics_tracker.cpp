#include "physics_tracker.h"
#include "config_tracking.h"
#include "algorithm_common.h"
#include "utils/conversions.h"
#include "utils_tracking.h"
#include <boost/thread.hpp>
#include "utils/testing.h"
#include "feature_extractor.h"
#include "utils/logging.h"
#include <cv.h>
#include <fstream>

using namespace Eigen;
using namespace std;

//#define CHECK_CORRECTNESS

PhysicsTracker::PhysicsTracker(TrackedObjectFeatureExtractor::Ptr object_features, FeatureExtractor::Ptr observation_features, VisibilityInterface::Ptr visibility_interface, const Eigen::MatrixXf& stdev) :
  m_objFeatures(object_features),
  m_obsFeatures(observation_features),
  m_visInt(visibility_interface)
{
  m_priorDist = m_objFeatures->m_obj->getPriorDist();
  m_stdev = stdev;

  m_outlierDist = m_objFeatures->m_obj->getOutlierDist();
}


PhysicsTracker::PhysicsTracker(TrackedObjectFeatureExtractor::Ptr object_features, FeatureExtractor::Ptr observation_features, VisibilityInterface::Ptr visibility_interface) :
  m_objFeatures(object_features),
  m_obsFeatures(observation_features),
  m_visInt(visibility_interface)
{
  m_priorDist = m_objFeatures->m_obj->getPriorDist();
  m_stdev = m_priorDist.transpose().replicate(m_objFeatures->m_obj->m_nNodes, 1);

  m_outlierDist = m_objFeatures->m_obj->getOutlierDist();
}

// Before calling this function, the inputs of the FeatureExtractors should be updated (if any)
void PhysicsTracker::updateFeatures() {
  m_objFeatures->updateFeatures();
  m_obsFeatures->updateFeatures();
  //shift the point cloud in the z coordinate
  //m_obsFeatures->getFeatures(FE::FT_XYZ).col(2) += VectorXf::Ones(m_obsFeatures->getFeatures(FE::FT_XYZ).rows()) * 0.01*METERS;
  
  for (int i=0; i<m_obsFeatures->getFeatures(FE::FT_XYZ).rows(); i++)
    if (m_obsFeatures->getFeatures(FE::FT_XYZ)(i,2) < 0.005*METERS)
      m_obsFeatures->getFeatures(FE::FT_XYZ)(i,2) = 0.005*METERS;


  m_estPts = m_objFeatures->getFeatures();
  m_obsPts = m_obsFeatures->getFeatures();  

  m_vis = m_visInt->checkNodeVisibility(m_objFeatures->m_obj);
}

void PhysicsTracker::expectationStep() {
  
  boost::posix_time::ptime e_time = boost::posix_time::microsec_clock::local_time();
  m_pZgivenC = calculateResponsibilities(m_estPts, m_obsPts, m_stdev, m_vis, m_objFeatures->m_obj->getOutlierDist(), m_objFeatures->m_obj->getOutlierStdev());

  LOG_DEBUG("E time " << (boost::posix_time::microsec_clock::local_time() - e_time).total_milliseconds());

#if 0
  boost::posix_time::ptime en_time = boost::posix_time::microsec_clock::local_time();
  MatrixXf pZgivenC_naive = calculateResponsibilitiesNaive(m_estPts, m_obsPts, m_stdev, m_vis, m_objFeatures->m_obj->getOutlierDist(), m_objFeatures->m_obj->getOutlierStdev());
  cout << "E naive time " << (boost::posix_time::microsec_clock::local_time() - en_time).total_milliseconds() << endl;
  cout << "Naive E Checking " << isApproxEq(pZgivenC_naive, m_pZgivenC) << endl;
#endif
}

void PhysicsTracker::maximizationStep(bool apply_evidence) {

  boost::posix_time::ptime evidence_time = boost::posix_time::microsec_clock::local_time();
  if (apply_evidence) m_objFeatures->m_obj->applyEvidence(m_pZgivenC, m_obsPts);
  LOG_DEBUG("Evidence time " << (boost::posix_time::microsec_clock::local_time() - evidence_time).total_milliseconds());

  boost::posix_time::ptime m_time = boost::posix_time::microsec_clock::local_time();
  m_stdev = calculateStdev(m_estPts, m_obsPts, m_pZgivenC, m_priorDist, TrackingConfig::pointPriorCount);
  LOG_DEBUG("M time " << (boost::posix_time::microsec_clock::local_time() - m_time).total_milliseconds());


#if 0
  //boost::posix_time::ptime evidence_time = boost::posix_time::microsec_clock::local_time();

  if (apply_evidence && isFinite(m_pZgivenC) && isFinite(m_estPts) && isFinite(m_obsPts))
    m_objFeatures->m_obj->applyEvidence(m_pZgivenC, m_obsFeatures->getFeatures(FE::FT_XYZ));
  //cout << "Evidence time " << (boost::posix_time::microsec_clock::local_time() - evidence_time).total_milliseconds() << endl;

  //boost::posix_time::ptime m_time = boost::posix_time::microsec_clock::local_time();
  if (isFinite(m_pZgivenC)) m_stdev = calculateStdev(m_estPts, m_obsPts, m_pZgivenC, m_priorDist, 1);
  //cout << "M time " << (boost::posix_time::microsec_clock::local_time() - m_time).total_milliseconds() << endl;
#endif

#if 0
  boost::posix_time::ptime mn_time = boost::posix_time::microsec_clock::local_time();
  MatrixXf stdev_naive = calculateStdevNaive(m_estPts, m_obsPts, m_pZgivenC, m_priorDist, 2);
  cout << "M naive time " << (boost::posix_time::microsec_clock::local_time() - mn_time).total_milliseconds() << endl;
  cout << "Naive M Checking " << isApproxEq(stdev_naive, m_stdev) << endl;
#endif
}


