#include "simplescene.h"
#include "util.h"
#include "rope.h"
#include "unistd.h"
#include <Eigen/Dense>
#include "dist_math.h"
#include <math.h>
#include "plotting.h"
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include "pcl/common/eigen.h"
#include <boost/foreach.hpp>
using boost::shared_ptr;
using namespace Eigen;
using namespace util;
// make table and rope from kinect data

const string data_dir = "/home/joschu/Data/pink_rope_sequence";

#define DEBUG_PLOTS

#ifdef DEBUG_PLOTS
PlotLines::Ptr forcelines;
PlotPoints::Ptr targpts;
#endif //DEBUG_PLOTS




void minRot(const btVector3& v1, const btVector3& v2, btMatrix3x3& m) {
  btScalar ang = v1.angle(v2);
  btVector3 ax = v2.cross(v1);
  m = btMatrix3x3(btQuaternion(ax,ang));

}

void verts2boxPars(const vector<btVector3>& verts, btVector3& halfExtents, btVector3& origin, btScalar thickness) {
  origin = (verts[0] + verts[2])/2;
  halfExtents = (verts[2] - verts[0]).absolute()/2;
  origin[2] -= thickness/2;
  halfExtents[0] *= 2;
  halfExtents[1] *= 2;
  halfExtents[2] = thickness/2;
}


MatrixXf toEigen(const vector<btVector3>& vecs) {
  MatrixXf out = MatrixXf(vecs.size(),3);
  for (int i=0; i < vecs.size(); i++) {
    //out.row(i).readArray(vecs[i].m_floats);
    out(i,0) = vecs[i].getX();
    out(i,1) = vecs[i].getY();
    out(i,2) = vecs[i].getZ();
  }
  return out;

}

void calcOptImpulses(const vector<btVector3>& ctrlPts, const vector<btVector3>& pointCloud, vector<btVector3>& forces) {
  MatrixXf ropePts = toEigen(ctrlPts);
  MatrixXf kinPts = toEigen(pointCloud);
  VectorXi indRope2Kin = argminAlongRows(pairwiseSquareDist(ropePts,kinPts));
  /*
  ofstream outfile("kinpts.txt");
  outfile << kinPts;
  outfile.close();
  ofstream outfile1("ropepts.txt");
  outfile1 << ropePts;
  outfile1.close();
  ofstream outfile2("indrope2kin.txt");
  outfile2 << indRope2Kin;
  outfile2.close();
  ofstream outfile3("sqdists.txt");
  outfile3 << pairwiseSquareDist(ropePts,kinPts);
  outfile3.close();
  cin.get();
  */

  vector<btVector3> plot_srcs;
  vector<btVector3> plot_targs;

  int nRope = ctrlPts.size();
  forces.resize(nRope);
  for (int i_rope=0; i_rope < nRope; i_rope++) {
    btVector3 cloud_pt = pointCloud[indRope2Kin[i_rope]];
    btVector3 rope_pt = ctrlPts[i_rope];
    forces[i_rope] = cloud_pt - rope_pt;
    plot_srcs.push_back(rope_pt);
    plot_targs.push_back(cloud_pt);
  }

  VectorXi indKin2Rope = argminAlongRows(pairwiseSquareDist(kinPts,ropePts));
  for (int i_cloud=0; i_cloud < indKin2Rope.size(); i_cloud++) {
    int i_rope = indKin2Rope[i_cloud];
    btVector3 rope_pt = ctrlPts[i_rope];
    btVector3 cloud_pt = pointCloud[i_cloud];
    forces[i_rope] += cloud_pt - rope_pt;
    plot_srcs.push_back(rope_pt);
    plot_targs.push_back(cloud_pt);
  }


#ifdef DEBUG_PLOTS
  forcelines->setPoints(plot_srcs,plot_targs);
  targpts->setPoints(pointCloud);
#endif //DEBUG_PLOTS

}

void applyImpulses(vector< shared_ptr<btRigidBody> > bodies, vector<btVector3> impulses, float multiplier) {
  for (int i=0; i<bodies.size(); i++) bodies[i]->applyCentralImpulse(impulses[i]*multiplier);
}

int main() {

  // load table plane coeffs
  // figure out the rotation matrix
  // load the rope
  // apply rotation to rope
  // make rope and add it to simulation




  ////////// files /////////////////////////////////////

  vector<string> pcdfiles;
  pcdfiles.push_back(data_dir+"/0001.pcd");
  pcdfiles.push_back(data_dir+"/0002.pcd");
  pcdfiles.push_back(data_dir+"/0003.pcd");
  pcdfiles.push_back(data_dir+"/0004.pcd");
  pcdfiles.push_back(data_dir+"/0005.pcd");
  pcdfiles.push_back(data_dir+"/0006.pcd");
  pcdfiles.push_back(data_dir+"/0007.pcd");
  pcdfiles.push_back(data_dir+"/0008.pcd");
  pcdfiles.push_back(data_dir+"/0009.pcd");
  pcdfiles.push_back(data_dir+"/0010.pcd");
  pcdfiles.push_back(data_dir+"/0011.pcd");

  vector<string> xyzfiles;
  xyzfiles.push_back(data_dir+"/0001_xyz.txt");
  xyzfiles.push_back(data_dir+"/0002_xyz.txt");
  xyzfiles.push_back(data_dir+"/0003_xyz.txt");
  xyzfiles.push_back(data_dir+"/0004_xyz.txt");
  xyzfiles.push_back(data_dir+"/0005_xyz.txt");
  xyzfiles.push_back(data_dir+"/0006_xyz.txt");
  xyzfiles.push_back(data_dir+"/0007_xyz.txt");
  xyzfiles.push_back(data_dir+"/0008_xyz.txt");
  xyzfiles.push_back(data_dir+"/0009_xyz.txt");
  xyzfiles.push_back(data_dir+"/00010_xyz.txt");
  xyzfiles.push_back(data_dir+"/00011_xyz.txt");




  ////////// figure out rope and table and transform //////////////////

  vector< vector<float> > xyzs_cam;
  read_2d_array(xyzs_cam,xyzfiles[0]); //rope vertices
  vector<float> abcd;
  read_1d_array(abcd, data_dir + "/coeffs.txt"); // table coefficients
  vector< vector<float> > verts_cam;
  read_2d_array(verts_cam,data_dir+"/verts.txt"); // table vertices

  btMatrix3x3 cam2world;

  btVector3 normal(abcd[0],abcd[1],abcd[2]);
  minRot(btVector3(0,0,1),-normal,cam2world);

  cout << "cam2world:" << endl;
  print_matrix<btMatrix3x3,3> (cam2world);
  btVector3 trans(0,0,1.5);
  btTransform cam2world1(cam2world,trans);


  vector<btVector3> verts_world;
  for (int i=0; i<verts_cam.size(); i++) verts_world.push_back(cam2world1*btVector3(verts_cam[i][0],verts_cam[i][1],verts_cam[i][2]));

  vector<btVector3> ctrlPts;
  for (int i=0; i<xyzs_cam.size(); i+=2) {
    btVector3 xyz_world = cam2world1*btVector3(xyzs_cam[i][0],xyzs_cam[i][1],xyzs_cam[i][2]);
    ctrlPts.push_back(xyz_world);
  }
  btVector3 halfExtents;
  btVector3 origin;

  verts2boxPars(verts_world,halfExtents,origin,.2);

  Affine3f T;
  Matrix3f M;
  for (int i=0; i<3; i++) for (int j=0; j<3; j++) M(i,j) = cam2world[i][j];
  Vector3f eig_trans(trans.getX(),trans.getY(),trans.getZ());
  T = Translation3f(eig_trans)*M;




  // const string pcdfile = "../data/0003.pcd";
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  // if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcdfile, *cloud) == -1) {
  //   PCL_ERROR(("couldn't read file " + pcdfile + "\n").c_str());
  //   return -1;
  // }
  // cout << "new cloud size: " << cloud->size();


  /////////////// put stuf into scene //////////////////

  shared_ptr<CapsuleRope> ropePtr(new CapsuleRope(ctrlPts,.01));
  shared_ptr<btMotionState> ms(new btDefaultMotionState(btTransform(btQuaternion(0,0,0,1),origin-btVector3(0,0,.01))));
  shared_ptr<BulletObject> table(new BoxObject(0,halfExtents,ms));
  Scene s = Scene(false,false,false);
  s.manip->state.debugDraw = false;
  PlotPoints::Ptr plot(new PlotPoints());
  s.env->add(plot);
  s.env->add(table);
  s.env->add(ropePtr);
#ifdef DEBUG_PLOTS
  forcelines.reset(new PlotLines(3));
  targpts.reset(new PlotPoints(20));
  s.env->add(forcelines);
  s.env->add(targpts);
#endif //DEBUG_PLOTS




  //////////////////// Declarations for stuff in loops

  vector<btVector3> pointCloud;
  vector<btVector3> forces(ctrlPts.size());
  vector<btVector3> centers;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;







  for (int t=0;  t<pcdfiles.size(); t++) {
    s.manip->toggleIdle();
    string pcdfile = pcdfiles[t];
    cout << "--------------- reading file " << pcdfile << "------------" << endl;
    string xyzfile = xyzfiles[t];
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcdfile, *cloud) == -1) {
      PCL_ERROR(("couldn't read file " + pcdfile + "\n").c_str());
      return -1;
    }
    cout << "new cloud size: " << cloud->size();
    pcl::transformPointCloud(*cloud,*cloud,T);
    plot->setPoints(cloud);
    s.step(0,1,.01);
    for (int i=0; i < 100; i++) {

      vector< vector<float> > pointCloud_;
      read_2d_array(pointCloud_,xyzfile);
      pointCloud.clear();
      BOOST_FOREACH(vector<float> vec,pointCloud_) pointCloud.push_back(cam2world1*btVector3(vec[0],vec[1],vec[2]));

      ropePtr->getPts(centers);
      calcOptImpulses(centers, pointCloud, forces);

      applyImpulses(ropePtr->bodies, forces, 3);
      s.step(.01,1,.01);
      cout << "iteration " << i  << endl;
      usleep(10*1000);
    }
  }
}
