add_library(btas2012alignment aligner.cpp transform.cpp)
target_link_libraries(btas2012alignment ${Boost_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(btas2012aligner main.cpp)
target_link_libraries(btas2012aligner btas2012alignment)
