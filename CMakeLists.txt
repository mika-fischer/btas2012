cmake_minimum_required(VERSION 2.8)
project(BTAS2012)

find_package(Boost COMPONENTS filesystem program_options system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(alignment)
