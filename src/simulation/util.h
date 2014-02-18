#ifndef __UTIL_H__
#define __UTIL_H__

#include <LinearMath/btTransform.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <openrave/openrave.h>
#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include "my_assert.h"
using namespace std;

#define STRINGIFY(x) #x
#define EXPAND(x) STRINGIFY(x)

void toggle(bool* b);
void add(int* n, int increment);

struct Scene;

namespace util {

  ///////////////// CONVERSIONS ////////////////////////////

  inline btTransform scaleTransform(const btTransform &t, btScalar a) {
      return btTransform(t.getRotation(), a*t.getOrigin());
  }
  inline btVector3 toBtVector(const OpenRAVE::Vector &v) { return btVector3(v.x, v.y, v.z); }
  inline btVector3 toBtVector(const Eigen::Vector3d &v) { return btVector3(v(0), v(1), v(2)); }
  inline btTransform toBtTransform(const OpenRAVE::Transform &t) {
    return btTransform(btQuaternion(t.rot.y, t.rot.z, t.rot.w, t.rot.x), toBtVector(t.trans));
  }
  inline btTransform toBtTransform(const OpenRAVE::Transform &t, btScalar scale) {
    return btTransform(btQuaternion(t.rot.y, t.rot.z, t.rot.w, t.rot.x), scale * toBtVector(t.trans));
  }

  inline OpenRAVE::Vector toRaveQuaternion(const btQuaternion &q) {
    btQuaternion q1 = q.normalized();
    return OpenRAVE::Vector(q1.w(), q1.x(), q1.y(), q1.z());
  }
  inline OpenRAVE::Vector toRaveVector(const btVector3 &v) {
    return OpenRAVE::Vector(v.x(), v.y(), v.z());
  }
  inline OpenRAVE::Transform toRaveTransform(const btTransform &t) {
    return OpenRAVE::Transform(toRaveQuaternion(t.getRotation()), toRaveVector(t.getOrigin()));
  }
  inline OpenRAVE::Transform toRaveTransform(const btTransform &t, btScalar scale) {
    return OpenRAVE::Transform(toRaveQuaternion(t.getRotation()), toRaveVector(scale * t.getOrigin()));
  }

  // Nan/Inf checking
  inline bool isfinite(const btVector3 &v) {
      return std::isfinite(v.x()) && std::isfinite(v.y()) && std::isfinite(v.z());
  }

  inline bool isfinite(const btMatrix3x3 &m) {
      return isfinite(m[0]) && isfinite(m[1]) && isfinite(m[2]);
  }

  inline bool isfinite(const btTransform &t) {
      return isfinite(t.getOrigin()) && isfinite(t.getBasis());
  }

  ///////////////// FILE IO ////////////////////////////
  template <class T>
  void read_2d_array(vector< vector<T> >& arr, string fname) {

    ifstream infile(fname.c_str());
    string line;
    arr.clear();
    while (getline(infile,line)) {
      stringstream ss (stringstream::in | stringstream::out);
      ss << line;
      vector<T> v;
      v.clear();
      while (ss) {
	T f;
	ss >> f;
	v.push_back(f);
      }
      arr.push_back(v);
    }
  }

  template <class T>
  void read_1d_array(vector<T>& arr, string fname) {
    ifstream infile(fname.c_str());
    T i;
    arr.clear();
    while (infile) {
      infile >>i;
      arr.push_back(i);
    }
  }



}

#endif // __UTIL_H__
