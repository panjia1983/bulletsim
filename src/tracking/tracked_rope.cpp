#include "tracked_object.h"
#include "config_tracking.h"
#include "utils/conversions.h"
#include "utils/cvmat.h"
#include "simulation/bullet_io.h"
#include "utils/utils_vector.h"
#include "feature_extractor.h"
#include <fstream>

using namespace std;
using namespace Eigen;

TrackedRope::TrackedRope(CapsuleRope::Ptr sim) : TrackedObject(sim, "rope") {
  m_nNodes = sim->children.size();
  m_masses.resize(m_nNodes);
  for (int i=0; i < m_nNodes; i++) {
    m_masses(i) = 1/getSim()->children[i]->rigidBody->getInvMass();
  }
}

std::vector<btVector3> TrackedRope::getPoints() {
	std::vector<btVector3> out(m_nNodes);
	for (int i=0; i < m_nNodes; ++i) {
          out[i] = getSim()->children[i]->rigidBody->getCenterOfMassPosition();
	}
	return out;
}

void TrackedRope::applyEvidence(const Eigen::MatrixXf& corr, const MatrixXf& obsPts) {
  vector<btVector3> estPos(m_nNodes), estVel(m_nNodes);
  for (int i=0; i < m_nNodes; ++i)  {
    estPos[i] = getSim()->children[i]->rigidBody->getCenterOfMassPosition();
    estVel[i] = getSim()->children[i]->rigidBody->getLinearVelocity();
  }


  std::ofstream obsPts_file("obsPts_offline.txt");
  std::ofstream estPos_file("estPos_offline.txt");
  std::ofstream corr_file("corr_offline.txt");
  std::ofstream estVel_file("estVel_offline.txt");


  for (int i = 0; i < obsPts.rows(); ++i) {
    for (int j = 0; j < obsPts.cols(); ++j) {
      obsPts_file << obsPts(i, j) << " ";
    }
    obsPts_file << endl;
  }

  for (int i = 0; i < estPos.size(); ++i) {
    estPos_file << estPos[i].x() << " " << estPos[i].y() << " " << estPos[i].z() << endl;
  }

  for (int i = 0; i < estVel.size(); ++i) {
    estVel_file << estVel[i].x() << " " << estVel[i].y() << " " << estVel[i].z() << endl;
  }

  for (int i = 0; i < corr.rows(); ++i) {
    for (int j = 0; j < corr.cols(); ++j) {
      corr_file << corr(i, j) << " ";
    }
    corr_file << endl;
  }

  cout << "parameter" << endl;
  cout << TrackingConfig::kp_rope << " " << TrackingConfig::kd_rope << endl;
  std::vector<float> masses = toVec(m_masses);
  for (int i = 0; i < masses.size(); ++i)
    cout << masses[i] << " ";
  cout << endl;
    


  vector<btVector3> impulses = calcImpulsesDamped(estPos, estVel, toBulletVectors(FE::activeFeatures2Feature(obsPts, FE::FT_XYZ)), corr, toVec(m_masses), TrackingConfig::kp_rope, TrackingConfig::kd_rope);

  cout << "impulses" << endl;
  for (int i = 0; i < impulses.size(); ++i) {
    cout << impulses[i].x() << " " << impulses[i].y() << " " << impulses[i].z() << endl;
  }


  for (int i=0; i<m_nNodes; ++i) getSim()->children[i]->rigidBody->applyCentralImpulse(impulses[i]);
}

void TrackedRope::initColors() {
	m_colors.resize(m_nNodes, 3);
	for (int i=0; i < m_nNodes; ++i) {
		Vector3f bgr = toEigenMatrixImage(getSim()->children[i]->getTexture()).colwise().mean();
		m_colors.row(i) = bgr.transpose();
	}
}
