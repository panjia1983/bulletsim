#pragma once
#include <ros/ros.h>
#include <bulletsim_msgs/ObjectInit.h>
#include <bulletsim_msgs/TrackedObject.h>
#include "utils_tracking.h"
#include "tracked_object.h"
#include "clouds/utils_pcl.h"
#include "utils_tracking.h"

bulletsim_msgs::TrackedObject toTrackedObjectMessage(TrackedObject::Ptr obj);
TrackedObject::Ptr callInitServiceAndCreateObject(ColorCloudPtr cloud, cv::Mat image, cv::Mat mask, CoordinateTransformer* transformer);
TrackedObject::Ptr callInitServiceAndCreateObject(const vector<btVector3>& nodes, float rope_radius, ColorCloudPtr cloud, cv::Mat image, cv::Mat mask, CoordinateTransformer* transformer);
