#include "simple_physics_tracker.h"
#include "sparse_utils.h"
#include "config_tracking.h"
#include "algorithm_common.h"
#include "utils/conversions.h"
#include "utils/clock.h"
#include <fstream>
#include <boost/thread.hpp>
#include "utils/testing.h"
#include "feature_extractor.h"
#include "utils/logging.h"

using namespace Eigen;
using namespace std;

//#define CHECK_CORRECTNESS

SimplePhysicsTracker::SimplePhysicsTracker(TrackedObject::Ptr obj, VisibilityInterface::Ptr visInt, Environment::Ptr env) :
  m_obj(obj),
  m_visInt(visInt),
  m_env(env),
  m_obsCloud(new ColorCloud()),
	m_applyEvidence(true),
	m_count(-1)
{
	m_obj_features = TrackedObjectFeatureExtractor::Ptr(new TrackedObjectFeatureExtractor(m_obj));
	m_cloud_features = CloudFeatureExtractor::Ptr(new CloudFeatureExtractor());

	m_prior_dist = m_obj->getPriorDist();
	m_stdev = m_prior_dist.transpose().replicate(m_obj->m_nNodes, 1);

	m_outlier_dist = m_obj->getOutlierDist();
}

void SimplePhysicsTracker::updateInput(ColorCloudPtr obsPts) {
	m_cloud_features->updateInputs(obsPts);

	m_obj_features->updateFeatures();
	m_cloud_features->updateFeatures();

	m_obsPts = m_cloud_features->getFeatures();
	//m_obsPts.col(2) = m_obsPts.col(2).array() + 0.05*METERS;
	for (int i=0; i<m_obsPts.rows(); i++) {
		if (m_obsPts(i,2) < 0.01*METERS) m_obsPts(i,2) = 0.01*METERS;
	}
	m_obsCloud = obsPts;
}

void SimplePhysicsTracker::doIteration() {
  VectorXf vis = m_visInt->checkNodeVisibility(m_obj);
  m_estPts = m_obj_features->getFeatures();

  // E STEP
  boost::posix_time::ptime e_time = boost::posix_time::microsec_clock::local_time();
  MatrixXf pZgivenC = calculateResponsibilities(m_estPts, m_obsPts, m_stdev, vis, m_obj->getOutlierDist(), m_obj->getOutlierStdev());
  LOG_DEBUG("E time " << (boost::posix_time::microsec_clock::local_time() - e_time).total_milliseconds());

#ifdef CHECK_CORRECTNESS
  boost::posix_time::ptime en_time = boost::posix_time::microsec_clock::local_time();
  MatrixXf pZgivenC_naive = calculateResponsibilitiesNaive(m_estPts, m_obsPts, m_stdev, vis, m_outlier_dist, m_prior_dist);
  cout << "E naive time " << (boost::posix_time::microsec_clock::local_time() - en_time).total_milliseconds() << endl;
  assert(isApproxEq(pZgivenC_naive, pZgivenC));
#endif

  VectorXf inlierFrac = pZgivenC.colwise().sum();
  // M STEP
  boost::posix_time::ptime evidence_time = boost::posix_time::microsec_clock::local_time();
  if (m_applyEvidence) m_obj->applyEvidence(pZgivenC, m_obsPts);
  LOG_DEBUG("Evidence time " << (boost::posix_time::microsec_clock::local_time() - evidence_time).total_milliseconds());

  boost::posix_time::ptime m_time = boost::posix_time::microsec_clock::local_time();
  m_stdev = calculateStdev(m_estPts, m_obsPts, pZgivenC, m_prior_dist, 10);
  LOG_DEBUG("M time " << (boost::posix_time::microsec_clock::local_time() - m_time).total_milliseconds());

#ifdef CHECK_CORRECTNESS
  boost::posix_time::ptime mn_time = boost::posix_time::microsec_clock::local_time();
  MatrixXf stdev_naive = calculateStdevNaive(m_estPts, m_obsPts, pZgivenC, m_prior_dist, 10);
  cout << "M naive time " << (boost::posix_time::microsec_clock::local_time() - mn_time).total_milliseconds() << endl;
  assert(isApproxEq(stdev_naive, m_stdev));
#endif

  m_env->step(.03,2,.015);
}
