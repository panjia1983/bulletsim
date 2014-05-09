#include "visibility.h"
#include "tracked_object.h"
#include "utils/utils_pcl.h"
#include "utils/conversions.h"
#include "simulation/config_bullet.h"
#include "utils/my_assert.h"
#include "simulation/bullet_io.h"
#include <fstream>

using namespace Eigen;

static const float DEPTH_OCCLUSION_DIST = .03;
static const float RAY_SHORTEN_DIST = .1;
static const float MIN_DEPTH = .4;

VectorXf EverythingIsVisible::checkNodeVisibility(TrackedObject::Ptr obj) {
  return VectorXf::Ones(obj->m_nNodes);
}


VectorXf DepthImageVisibility::checkNodeVisibility(TrackedObject::Ptr obj) {
  MatrixXf ptsCam = toEigenMatrix(m_transformer->toCamFromWorldN(obj->getPoints()));
  VectorXf ptDists = ptsCam.rowwise().norm();
  MatrixXi uvs = xyz2uv(ptsCam);
  VectorXf vis(ptsCam.rows());
  for (int i = 0; i < vis.size(); ++i)
    vis(i) = 1.0;

  cout << "ptDists" << endl;
  for(int i = 0; i < ptDists.size(); ++i)
    cout << ptDists(i) << " ";
  cout << endl;
  std::ofstream depth_file("depth_online.txt");
  for (int i = 0; i < m_depth.rows; ++i) {
    for (int j = 0; j < m_depth.cols; ++j) {
      depth_file << m_depth.at<float>(i, j) << " ";
    }
    depth_file << endl;
  }

  assert(m_depth.type() == CV_32FC1);
  float occ_dist = DEPTH_OCCLUSION_DIST*METERS;

  for (int iPt=0; iPt<ptsCam.rows(); ++iPt) {
    int u = uvs(iPt,0);
    int v = uvs(iPt,1);
    if (u<m_depth.rows && v<m_depth.cols && u>0 && v>0) {
      bool is_vis = !isfinite(m_depth.at<float>(u,v)) || (m_depth.at<float>(u,v) + occ_dist > ptDists[iPt]);
      if (is_vis)
        vis(iPt) = 1.0;
      else 
        vis(iPt) = 0.0;
    // see it if there's no non-rope pixel in front of it
    }
  }

  return vis;
}

void DepthImageVisibility::updateInput(const cv::Mat& in) {
	m_depth = in;
}


BulletRaycastVisibility::BulletRaycastVisibility(btDynamicsWorld* world, CoordinateTransformer* transformer)
	: m_world(world), m_transformer(transformer) {
}

#ifdef PLOT_RAYCAST
#include "plotting_tracking.h"
#endif

VectorXf BulletRaycastVisibility::checkNodeVisibility(TrackedObject::Ptr obj) {
	vector<btVector3> nodes = obj->getPoints();
	btVector3 cameraPos = m_transformer->worldFromCamUnscaled.getOrigin()*METERS;
	VectorXf vis(nodes.size());

	float ray_shorten_dist = RAY_SHORTEN_DIST*METERS;
	float min_depth = MIN_DEPTH*METERS;

#ifdef PLOT_RAYCAST
    static PlotLines::Ptr rayTestLines;
    static PlotPoints::Ptr rayHitPoints;
    if (!rayTestLines) {
      rayTestLines.reset(new PlotLines(4));
      rayHitPoints.reset(new PlotPoints(30));
      getGlobalEnv()->add(rayTestLines);
      rayTestLines->setColor(1,1,1,1);
      printf("adding ray test lines to env\n");
    }
    vector<btVector3> linePoints;
    vector<btVector3> hitPoints;
#endif

	for (int i=0; i < nodes.size(); ++i) {
	  btVector3 rayDir = (nodes[i] - cameraPos).normalized();
		btVector3 rayEnd = nodes[i] - ray_shorten_dist * rayDir;
		btVector3 rayStart = cameraPos + min_depth * rayDir;
		btCollisionWorld::ClosestRayResultCallback rayCallback(rayStart, rayEnd);
		m_world->rayTest(rayStart, rayEnd, rayCallback);
		vis[i] = !rayCallback.hasHit();

#ifdef PLOT_RAYCAST
		if (rayCallback.hasHit()) hitPoints.push_back(rayCallback.m_hitPointWorld);
		linePoints.push_back(rayStart);
		linePoints.push_back(rayEnd);
#endif
	}
#ifdef PLOT_RAYCAST
	rayTestLines->setPoints(linePoints);
#endif
	return vis;
}





