#pragma once

#include "config.h"
#include "plotting.h"

#include <btBulletDynamicsCommon.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <iostream>
#include <boost/shared_ptr.hpp>

using namespace std;
using namespace Eigen;
using boost::shared_ptr;

vector<bool> checkOccluded(const vector<btVector3>& xyzs_world, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,  Affine3f cam2world);
MatrixXi xyz2uv(const Matrix<float,Dynamic,3>& xyz);
Affine3f getCam2World(const vector<Vector3f>& tableVerts, Vector3f& normal, float scale=1);
MatrixXi xyz2uv(const Matrix<float,Dynamic,3>& xyz);
void calcOptImpulses(const vector<btVector3>& ctrlPts, const vector<btVector3>& pointCloud, const vector<btVector3>& oldCtrlPts, vector<btVector3>& forces, vector<bool>& occs);
void applyImpulses(vector< shared_ptr<btRigidBody> > bodies, vector<btVector3> impulses, float multiplier);
void initTrackingPlots();

struct plots {
  static PlotLines::Ptr forcelines;
  static PlotPoints::Ptr targpts;
};





