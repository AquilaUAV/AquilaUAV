cmake_minimum_required(VERSION 3.14)
project(aquila_cpp)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 14)

add_executable(aquila_cpp main.cpp)
target_link_libraries( aquila_cpp ${OpenCV_LIBS} )