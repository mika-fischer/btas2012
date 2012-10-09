cmake_minimum_required(VERSION 2.8)
project(BTAS2012)

set(Boost_USE_STATIC_LIBS TRUE)
find_package(Boost COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})
add_definitions(-DBOOST_ALL_NO_LIB)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(alignment)
