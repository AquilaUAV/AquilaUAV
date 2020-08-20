И так, поехали.
Источник:
https://github.com/gsilano/BebopS
Задача - собрать переделать под ROS noetic.
...

Новые инструкции установки:

```bash
... stuff ...

sudo apt install ros-noetic-desktop-full
source /opt/ros/noetic/setup.bash

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
catkin_init_workspace  # initialize your catkin workspace
cd ~/catkin_ws/
catkin init
cd ~/catkin_ws/src
git clone -b med18_gazebo9 https://github.com/gsilano/rotors_simulator.git
git clone -b med18_gazebo9 https://github.com/gsilano/mav_comm.git
git clone -b dev/gazebo9 https://github.com/gsilano/BebopS.git
git clone https://github.com/AutonomyLab/bebop_autonomy.git
git clone https://github.com/AutonomyLab/parrot_arsdk.git
cd ~/catkin_ws

sudo apt install -y ros-noetic-octomap ros-noetic-octomap-msgs ros-noetic-octomap-ros
sudo apt install -y libgoogle-glog-dev
sudo apt install -y python # python2-is-python... [На этом месты было потрачено 40 минут, вот я turtlebot...]
sudo apt install -y libavahi-client-dev

#"""Далее будет проблемы либ"""
#1 :: in file ~/catkin_ws/src/rotors_simulator/rotors_gazebo/CMakeLists.txt
cmake_minimum_required(VERSION 2.8.3) -> cmake_minimum_required(VERSION 3.13)

#2 :: in file ~/catkin_ws/src/rotors_simulator/rotors_gazebo/CMakeLists.txt
"hovering_example" -> "rotors_hovering_example" 

catkin_make
catkin_make
catkin_make

# Остались только косяки со старым OpenCV, мы то на 4.2.0+ сидим.

# 4. :: in file ~/catkin_ws/src/rotors_simulator/rotors_gazebo_plugins/src/external/gazebo_geotagged_images_plugin.cpp
"#include <cv.h>" -> "//#include <cv.h>" 
"#include <highgui.h>" -> "#include "opencv2/highgui/highgui.hpp""
"cvtColor(frame, frameBGR, CV_RGB2BGR);" -> "cvtColor(frame, frameBGR, COLOR_RGB2BGR);"

# 5. :: in file ~/catkin_ws/src/rotors_simulator/rotors_gazebo_plugins/src/gazebo_odometry_plugin.cpp
"CV_LOAD_IMAGE_GRAYSCALE" -> "cv::IMREAD_GRAYSCALE"

# И после всего этого гемора оно билдится :D

#""""""

```

