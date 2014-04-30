include_directories( ${PCL_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -Wno-enum-compare -fvisibility=default")

add_library(bulletUtils config.cpp conversions.cpp logging.cpp utils_vector.cpp cvmat.cpp testing.cpp clock.cpp utils_pcl.cpp)
target_link_libraries(bulletUtils ${LOG4CPLUS_LIBRARY} ${BOOST_LIBRARIES} ${BULLET_LIBS} ${OpenCV_LIBS} ${PCL_LIBRARIES}) 