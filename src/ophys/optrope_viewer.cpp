#include "optrope.h"

#include "ophys_config.h"
#include "ophys_common.h"

#include "simulation/simplescene.h"
#include "simulation/plotting.h"
#include "robots/pr2.h"

using namespace Eigen;
using namespace std;

using namespace ophys;


struct LocalConfig : public Config {
  static string in;
  static int extraInterp;
  static bool noRopeLine;
  static double ptRadius;
  LocalConfig() : Config() {
    params.push_back(new Parameter<string>("in", &in, "input file"));
    params.push_back(new Parameter<int>("extraInterp", &extraInterp, ""));
    params.push_back(new Parameter<bool>("noRopeLine", &noRopeLine, ""));
    params.push_back(new Parameter<double>("ptRadius", &ptRadius, ""));
  }
};
string LocalConfig::in = "";
int LocalConfig::extraInterp = 5;
bool LocalConfig::noRopeLine = false;
double LocalConfig::ptRadius = 0.01;


struct OptRopePlot {
  Scene *m_scene;
  PlotSpheres::Ptr m_plotSpheres;
  PlotLines::Ptr m_plotLines;
  const int m_N;
  bool m_useRobot;
  RobotManipulator::Ptr m_manip;
  MatrixX3d m_initPos;
  Vector7d m_initManipDofs;
  int m_extraInterp;

  OptRopePlot(int N, int extraInterp, Scene *scene, const MatrixX3d &initPos, const Vector7d &initManipDofs)
    : m_N(N),
      m_scene(scene),
      m_plotSpheres(new PlotSpheres),
      m_plotLines(new PlotLines(5)),
      m_useRobot(false),
      m_initPos(initPos), m_initManipDofs(initManipDofs),
      m_extraInterp(extraInterp)
  {
    m_scene->env->add(m_plotSpheres);
    m_scene->env->add(m_plotLines);
    draw(initPos, initManipDofs);
  }

  void setRobot(RobotManipulator::Ptr manip) {
    m_useRobot = true;
    m_manip = manip;
  }

  void draw(const MatrixX3d &pos, const Vector7d &manipDofs) {
    // rope control points
    osg::ref_ptr<osg::Vec3Array> centers(new osg::Vec3Array());
    osg::ref_ptr<osg::Vec4Array> rgba(new osg::Vec4Array());
    vector<float> radii;
    for (int i = 0; i < pos.rows(); ++i) {
      centers->push_back(osg::Vec3(pos(i, 0), pos(i, 1), pos(i, 2)));
      double a = (double)i/(double)pos.rows();
      rgba->push_back(osg::Vec4(a, 0, 1.-a, 1));
      radii.push_back(LocalConfig::ptRadius);
    }

    // special point
    //Vector3d p = centroid(m_initPos) + Vector3d(0.1, 0, 0.1);
    // Vector3d p = m_initPos.row(0).transpose() + Vector3d(0.2, -.2, 0.2);//   m_initPos.row(0).transpose() + Vector3d(0.2, 0, 0.2);
    // centers->push_back(osg::Vec3(p[0], p[1], p[2]));
    // rgba->push_back(osg::Vec4(1, 0, 0, 1));
    // radii.push_back(0.02);

    // manipulator position
    if (!m_useRobot) {
      centers->push_back(osg::Vec3(manipDofs(0), manipDofs(1), manipDofs(2)));
      rgba->push_back(osg::Vec4(0, 1, 0, 0.7));
      radii.push_back(0.03);
    }

    m_plotSpheres->plot(centers, rgba, radii);

    // imaginary lines connecting rope control points
    if (!LocalConfig::noRopeLine) {
      osg::ref_ptr<osg::Vec3Array> linePts(new osg::Vec3Array());
      for (int i = 0; i < pos.rows() - 1; ++i) {
        linePts->push_back(osg::Vec3(pos(i, 0), pos(i, 1), pos(i, 2)));
        linePts->push_back(osg::Vec3(pos(i+1, 0), pos(i+1, 1), pos(i+1, 2)));
      }
      m_plotLines->setPoints(linePts);
    }

    if (m_useRobot) {
      m_manip->setDOFValues(toStlVec(manipDofs));
      btVector3 p = m_manip->getTransform().getOrigin();
      // cout << "current end effector pos: " << p.x() << ' ' << p.y() << ' ' << p.z() << endl;
    }
  }

  void playTraj(const OptRopeState &s, bool idlePerStep=false, bool printProgress=false) {
    MatrixX3d x(s.atTime[0].x.rows(), 3);
    Vector7d manipDofs;

    for (int t = 0; t < s.atTime.size(); ++t) {
      if (printProgress) {
        cout << "showing step " << (t+1) << "/" << s.atTime.size() << endl;
      }
      for (int i = 0; i < m_extraInterp; ++i) {
        if (t < s.atTime.size() - 1) {
          double a = (double)i/(double)m_extraInterp;
          x = (1.-a)*s.atTime[t].x + a*s.atTime[t+1].x;
          manipDofs = (1.-a)*s.atTime[t].manipDofs + a*s.atTime[t+1].manipDofs;
        } else {
          x = s.atTime[t].x;
          manipDofs = s.atTime[t].manipDofs;
        }
        draw(x, manipDofs);
        m_scene->step(0);
        m_scene->idleFor(0.01);
        //if (idlePerStep) m_scene->idle(true);
      }
    }
  }
};


static vector<btVector3> initTableCornersWorld() {
  vector<btVector3> v;
  v.push_back(btVector3(OPhysConfig::tableDistFromRobot, -OPhysConfig::tableWidth/2, OPhysConfig::tableHeight));
  v.push_back(btVector3(OPhysConfig::tableDistFromRobot, OPhysConfig::tableWidth/2, OPhysConfig::tableHeight));
  v.push_back(btVector3(OPhysConfig::tableDistFromRobot + OPhysConfig::tableLength, OPhysConfig::tableWidth/2, OPhysConfig::tableHeight));
  v.push_back(btVector3(OPhysConfig::tableDistFromRobot + OPhysConfig::tableLength, -OPhysConfig::tableWidth/2, OPhysConfig::tableHeight));
  return v;
}

static BulletObject::Ptr makeTable(const vector<btVector3>& corners, float thickness) {
  btVector3 origin = (corners[0] + corners[2])/2;
  origin[2] -= thickness/2;
  btVector3 halfExtents = (corners[2] - corners[0]).absolute()/2;
  halfExtents[2] = thickness/2;

  return BulletObject::Ptr(new BoxObject(0,halfExtents,btTransform(btQuaternion(0,0,0,1),origin)));
}


int main(int argc, char *argv[]) {
  Parser parser;
  parser.addGroup(LocalConfig());
  parser.addGroup(OPhysConfig());
  parser.read(argc, argv);

  // read file to play
  OptRopeState state = OptRopeState::ReadFromFile(LocalConfig::in);
  cout << state.toString() << endl;

  // set up scene
  Scene scene;
  boost::shared_ptr<PR2Manager> pr2m;
  if (OPhysConfig::useRobot) {
    pr2m.reset(new PR2Manager(scene));

    // put the robot in a nice initial pose
    vector<int> indices(1, pr2m->pr2->robot->GetJointIndex("torso_lift_joint"));
    vector<double> values(1, .31);
    pr2m->pr2->setDOFValues(indices, values);
    pr2m->pr2Left->setDOFValues(pr2LeftNeutralPos());
    //pr2m->pr2Right->setDOFValues(pr2RightNeutralPos());
    pr2m->pr2Right->setDOFValues(toStlVec(state.atTime[0].manipDofs));
  }
  BulletObject::Ptr table = makeTable(initTableCornersWorld(), 0.01);
  scene.env->add(table);
  table->setColor(0, 1, 0, 0.2);

  OptRopePlot plot(state.m_N, LocalConfig::extraInterp, &scene, state.atTime[0].x, state.atTime[0].manipDofs);
  if (OPhysConfig::useRobot) {
    plot.setRobot(pr2m->pr2Right);
  }

  scene.startViewer();
  scene.idle(true);
  plot.playTraj(state, true, true);
  scene.idle(true);

  return 0;
}
