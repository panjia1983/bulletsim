include_directories(
    ${BULLET_DIR}/Extras
    ${BULLET_DIR}/Extras/HACD
    ${BULLETSIM_SOURCE_DIR}/lib/haptics
    ${BULLETSIM_SOURCE_DIR}/src
    ${TETGEN_DIR}
    ${PCL_INCLUDE_DIRS} 
)

#SET(CMAKE_CXX_FLAGS "-Wall -Wno-sign-compare -Wno-reorder")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-enum-compare -Wno-attributes -Wno-unused-result -fvisibility=default")


add_library(bulletTracking
  algorithm_common.cpp   
  sparse_utils.cpp      
  visibility.cpp
  bulletrac.cpp          
  surface_sampling.cpp
  config_tracking.cpp    
  physics_tracker.cpp         
  utils_cv.cpp
  feature_extractor.cpp  
  simple_physics_tracker.cpp  
  utils_tracking.cpp
  tracked_object.cpp
  tracked_rope.cpp
)

target_link_libraries(bulletTracking
  bulletSimulation
  bulletUtils
  ${Boost_LIBRARIES}
  ${BULLET_LIBS}
  ${OpenRAVE_LIBRARIES}
  ${OpenRAVE_CORE_LIBRARIES}
  ${LOG4CPLUS_LIBRARY}
)

boost_python_module(cbulletracpy bulletracpy.cpp)
target_link_libraries(cbulletracpy bulletUtils bulletSimulation bulletTracking)
