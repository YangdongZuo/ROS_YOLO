export TURTLEBOT3_MODEL=waffle_pi
source ~/.bashrc

roslaunch ros_autonomous_slam turtlebot3_world.launch
roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
rosrun map_server map_saver -f my_map

roslaunch ros_autonomous_slam turtlebot3_navigation.launch
